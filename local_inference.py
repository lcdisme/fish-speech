#!/usr/bin/env python3
"""
Fish-Speech 本地推理脚本
用于中文说书类视频的语音合成，支持语音克隆和长文本切分

用法示例:
    # 基础语音合成（使用默认参考语音）
    python local_inference.py --text "大家好，欢迎收听今天的说书节目。" --output output.wav
    
    # 语音克隆
    python local_inference.py --text "从前有座山，山上有座庙。" \
        --ref_audio ref.wav --prompt_text "这是参考音频" --output output.wav
    
    # 长文本 + SRT 字幕
    python local_inference.py --text-file story.txt --output story.wav --output_srt story.srt
"""

import argparse
import gc
import os
import re
import sys
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import pyrootutils
import soundfile as sf
import torch
import torchaudio
from loguru import logger

# Setup project root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

# 默认参考语音和文字路径
DEFAULT_REF_PATH = Path(__file__).parent / "ref_voice" / "ref_cut.wav"
DEFAULT_PROMPT_TEXT_PATH = Path("/Users/lichengdou/PycharmProjects/pythonProject/fish_speech/ref_voice/ref_text.txt")

# 读取默认参考文字
def _load_default_prompt_text():
    if DEFAULT_PROMPT_TEXT_PATH.exists():
        with open(DEFAULT_PROMPT_TEXT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

DEFAULT_PROMPT_TEXT = _load_default_prompt_text()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fish-Speech 本地推理脚本 - 支持语音克隆和长文本切分",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    # 文本输入
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text", "-t",
        type=str,
        help="要合成的文本内容"
    )
    text_group.add_argument(
        "--text-file", "-tf",
        type=Path,
        help="包含要合成文本的文件路径"
    )
    
    # 输出设置
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output.wav"),
        help="输出音频文件路径 (默认: output.wav)"
    )
    
    # 参考音频（语音克隆）- 与 tts 项目接口一致
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=str(DEFAULT_REF_PATH),
        help="参考音频文件路径（用于语音克隆），默认使用 ref_voice/ref_cut.wav"
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=DEFAULT_PROMPT_TEXT,
        help="参考音频对应的文字内容，默认从 ref_voice/ref_text.txt 读取"
    )
    
    # 文本切分设置
    parser.add_argument(
        "--max-chars", "-mc",
        type=int,
        default=200,
        help="每段文本最大字符数 (默认: 200)"
    )
    
    # 字幕生成 - 与 tts 项目接口一致
    parser.add_argument(
        "--output_srt",
        type=str,
        default=None,
        help="输出 SRT 字幕文件路径（可选，默认不生成）"
    )
    
    # 句尾停顿设置
    parser.add_argument(
        "--sentence-pause",
        type=float,
        default=0.5,
        help="每个句子结束后添加的静音时长（秒），默认 0 表示不添加"
    )
    
    # 字幕最大字符数
    parser.add_argument(
        "--subtitle-max-chars",
        type=int,
        default=23,
        help="每行字幕最大字符数（默认 23，确保视频字幕一行显示）"
    )
    parser.add_argument(
        "--subtitle-min-chars",
        type=int,
        default=3,
        help="每行字幕最小字符数（默认 3，过短会合并到下一行）"
    )
    
    # 模型设置
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoints/openaudio-s1-mini"),
        help="模型检查点目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="运行设备 (auto 自动检测)"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="启用 torch.compile 加速（首次运行较慢）"
    )
    
    # 生成参数
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="温度参数 (默认: 0.7, 越低越稳定)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p 采样 (默认: 0.8)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="重复惩罚 (默认: 1.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于复现结果）"
    )
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """自动检测最佳设备"""
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            logger.info("检测到 Apple Silicon MPS，使用 MPS 加速")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("检测到 CUDA，使用 GPU 加速")
            return "cuda"
        else:
            logger.info("使用 CPU 进行推理")
            return "cpu"
    return device_arg


def split_text_smart(text: str, max_chars: int = 200, keep_sentences_separate: bool = True) -> List[str]:
    """
    智能切分长文本
    - 按句子边界切分（。！？；）
    - 确保每段不超过 max_chars
    - keep_sentences_separate=True 时，每个句子独立（适合添加句尾停顿）
    - keep_sentences_separate=False 时，合并短句以减少片段数
    """
    # 清理文本
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # 合并多余空白
    
    # 按句子边界切分
    sentence_delimiters = r'([。！？；!?;])'
    parts = re.split(sentence_delimiters, text)
    
    # 重新组合句子（保留标点）
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i]
        if i + 1 < len(parts):
            sentence += parts[i + 1]
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)
    
    # 处理可能残余的最后部分
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    
    # 如果没有句子边界，按逗号切分
    if len(sentences) <= 1 and len(text) > max_chars:
        parts = re.split(r'([，,])', text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i + 1]
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())
    
    # 如果 keep_sentences_separate=True，直接返回句子列表（只处理超长句子）
    if keep_sentences_separate:
        segments = []
        for sentence in sentences:
            if len(sentence) > max_chars:
                # 超长句子强制切分
                for i in range(0, len(sentence), max_chars):
                    segments.append(sentence[i:i + max_chars])
            else:
                segments.append(sentence)
        return [s.strip() for s in segments if s.strip()]
    
    # 合并短句子，确保每段不超过 max_chars
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        # 如果单句已超过限制，强制切分
        if len(sentence) > max_chars:
            if current_segment:
                segments.append(current_segment)
                current_segment = ""
            # 按字符数强制切分
            for i in range(0, len(sentence), max_chars):
                segments.append(sentence[i:i + max_chars])
        elif len(current_segment) + len(sentence) <= max_chars:
            current_segment += sentence
        else:
            if current_segment:
                segments.append(current_segment)
            current_segment = sentence
    
    if current_segment:
        segments.append(current_segment)
    
    # 过滤空段落
    segments = [s.strip() for s in segments if s.strip()]
    
    return segments


def split_subtitle(text: str, start_time: float, end_time: float, max_chars: int = 23, min_chars: int = 3) -> List[dict]:
    """
    智能拆分字幕
    规则:
    1. 按标点符号(，。！？；)切分
    2. 每行最少 min_chars 个字，不够则与下一行合并
    3. 每行最多 max_chars 个字，超出则强制换行
    4. 移除所有标点符号
    """
    text = text.strip()
    if not text:
        return []
    
    # 1. 按指定标点符号切分
    # 包含中英文标点
    split_pattern = r'([，。！？；,.!?;])'
    raw_parts = re.split(split_pattern, text)
    
    # 过滤掉标点符号，只保留文本部分作为基础原子
    # 注意：这里我们直接丢弃了标点符号，满足"最后生成的所有字幕再去掉标点符号"的需求
    atoms = []
    for part in raw_parts:
        clean_part = part.strip()
        # 排除纯标点或空字符串
        if clean_part and not re.match(r'^[，。！？；,.!?;]+$', clean_part):
            atoms.append(clean_part)
            
    if not atoms:
        return []

    # 2. 合并过短的片段 (Min Length Constraint)
    merged_atoms = []
    if len(atoms) > 0:
        current_atom = atoms[0]
        for next_atom in atoms[1:]:
            # 如果当前片段过短，尝试合并下一个
            if len(current_atom) < min_chars:
                current_atom += next_atom
            else:
                merged_atoms.append(current_atom)
                current_atom = next_atom
        merged_atoms.append(current_atom)
    else:
        merged_atoms = atoms

    # 如果最后一段仍然过短且前面有内容，尝试合并到前一段
    if len(merged_atoms) > 1 and len(merged_atoms[-1]) < min_chars:
        last = merged_atoms.pop()
        merged_atoms[-1] += last

    # 3. 处理过长的片段 (Max Length Constraint)
    final_chunks = []
    for atom in merged_atoms:
        if len(atom) <= max_chars:
            final_chunks.append(atom)
        else:
            # 强制切分
            for i in range(0, len(atom), max_chars):
                final_chunks.append(atom[i:i + max_chars])
    
    # 再次清理可能残留的空字符串，并移除句中残留的标点符号
    cleaned_chunks = []
    # 匹配所有常见中英文标点符号
    punct_pattern = r'[、`~@#$%^&*()_+\-=\[\]{}\\|<>/！!？?。.,，;："\'“”‘’《》…—]+'
    
    for c in final_chunks:
        # 将标点替换为空
        cleaned = re.sub(punct_pattern, '', c).strip()
        if cleaned:
            cleaned_chunks.append(cleaned)
    
    if not cleaned_chunks:
        return []

    # 4. 根据字符数分配时间
    total_chars = sum(len(c) for c in cleaned_chunks)
    duration = end_time - start_time
    
    result = []
    current_time = start_time
    
    for chunk in cleaned_chunks:
        # 简单的按字符比例分配时间
        chunk_len = len(chunk)
        chunk_duration = duration * (chunk_len / total_chars) if total_chars > 0 else 0
        
        result.append({
            'text': chunk,
            'start': current_time,
            'end': current_time + chunk_duration
        })
        current_time += chunk_duration
        
    return result


def generate_srt(segments: List[str], segment_durations: List[float], srt_path: Path, max_chars: int = 23, min_chars: int = 3):
    """生成 SRT 字幕文件，自动拆分长字幕"""
    
    current_time = 0.0
    srt_content = []
    subtitle_index = 1
    
    for text, duration in zip(segments, segment_durations):
        start_time = current_time
        end_time = current_time + duration
        
        # 拆分长字幕
        sub_items = split_subtitle(text, start_time, end_time, max_chars, min_chars)
        
        for item in sub_items:
            start_str = format_srt_time(item['start'])
            end_str = format_srt_time(item['end'])
            
            srt_content.append(f"{subtitle_index}")
            srt_content.append(f"{start_str} --> {end_str}")
            srt_content.append(item['text'])
            srt_content.append("")
            subtitle_index += 1
        
        current_time = end_time
    
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_content))
    
    logger.info(f"SRT 字幕已保存: {srt_path}，共 {subtitle_index - 1} 条")


def format_srt_time(seconds: float) -> str:
    """格式化时间为 SRT 格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def load_reference_audio(audio_path: Path, sample_rate: int = 44100) -> bytes:
    """加载参考音频并转换为 bytes"""
    import io
    import wave
    
    # 使用 soundfile 加载音频（避免 torchaudio 的 torchcodec 依赖问题）
    audio_np, sr = sf.read(str(audio_path))
    
    # 转为单声道
    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)
    
    # 重采样到目标采样率
    if sr != sample_rate:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=sample_rate)
    
    # 转换为 16-bit PCM bytes
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())
    
    return buffer.getvalue()


class LocalTTSEngine:
    """本地 TTS 引擎封装"""
    
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        compile: bool = False,
    ):
        self.device = get_device(device)
        self.compile = compile
        self.checkpoint_path = checkpoint_path
        
        # 设置精度
        self.precision = torch.bfloat16
        if self.device == "mps":
            # MPS 对 bfloat16 支持有限，使用 float16
            self.precision = torch.float16
        
        logger.info(f"初始化 TTS 引擎 (设备: {self.device}, 精度: {self.precision})")
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """加载 LLAMA 和 Decoder 模型"""
        llama_path = self.checkpoint_path
        decoder_path = self.checkpoint_path / "codec.pth"
        
        logger.info("正在加载 LLAMA 模型...")
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(llama_path),
            device=self.device,
            precision=self.precision,
            compile=self.compile,
        )
        
        logger.info("正在加载 Decoder 模型...")
        self.decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(decoder_path),
            device=self.device,
        )
        
        # 创建推理引擎
        self.engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=self.compile,
        )
        
        logger.info("模型加载完成")
    
    def synthesize(
        self,
        text: str,
        reference_audios: Optional[List[bytes]] = None,
        reference_texts: Optional[List[str]] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        合成单段音频
        返回: (audio_array, sample_rate)
        """
        # 构建参考音频列表
        references = []
        if reference_audios and reference_texts:
            for audio, text_ref in zip(reference_audios, reference_texts):
                references.append(ServeReferenceAudio(audio=audio, text=text_ref))
        
        # 构建请求
        request = ServeTTSRequest(
            text=text,
            references=references,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            chunk_length=200,
            max_new_tokens=1024,
            format="wav",
            streaming=False,
        )
        
        # 执行推理
        audio_data = None
        sample_rate = 44100
        
        for result in self.engine.inference(request):
            if result.code == "final":
                sample_rate, audio_data = result.audio
                break
            elif result.code == "error":
                raise RuntimeError(f"推理失败: {result.error}")
        
        if audio_data is None:
            raise RuntimeError("未生成音频")
        
        return audio_data, sample_rate
    
    def synthesize_long(
        self,
        text: str,
        max_chars: int = 200,
        reference_audios: Optional[List[bytes]] = None,
        reference_texts: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[Tuple[str, np.ndarray, int], None, None]:
        """
        长文本分段合成
        Yields: (segment_text, audio_array, sample_rate)
        """
        segments = split_text_smart(text, max_chars)
        total = len(segments)
        
        logger.info(f"文本已切分为 {total} 段")
        
        for i, segment in enumerate(segments, 1):
            logger.info(f"正在合成第 {i}/{total} 段: {segment[:30]}...")
            
            try:
                audio, sr = self.synthesize(
                    text=segment,
                    reference_audios=reference_audios,
                    reference_texts=reference_texts,
                    **kwargs
                )
                yield segment, audio, sr
                
                # 清理 MPS/CUDA 缓存
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"合成第 {i} 段失败: {e}")
                raise


def main():
    args = parse_args()
    
    # 读取文本
    if args.text:
        text = args.text
    else:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    
    logger.info(f"输入文本长度: {len(text)} 字符")
    
    # 初始化引擎
    engine = LocalTTSEngine(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        compile=args.compile,
    )
    
    # 加载参考音频
    ref_audio_path = Path(args.ref_audio)
    prompt_text = args.prompt_text
    
    reference_audios = None
    reference_texts = None
    
    if ref_audio_path.exists():
        logger.info(f"加载参考音频: {ref_audio_path}")
        reference_audios = [load_reference_audio(ref_audio_path)]
        reference_texts = [prompt_text]
        logger.info(f"参考文字: {prompt_text[:50]}...")
    else:
        logger.warning(f"参考音频不存在: {ref_audio_path}，将使用零样本合成")
    
    # 合成音频
    start_time = time.time()
    
    all_audios = []
    all_segments = []
    segment_durations = []
    sample_rate = 44100
    
    for segment_text, audio, sr in engine.synthesize_long(
        text=text,
        max_chars=args.max_chars,
        reference_audios=reference_audios,
        reference_texts=reference_texts,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    ):
        # 在句子结束处添加静音（如果设置了 sentence-pause）
        if args.sentence_pause > 0:
            last_char = segment_text.rstrip()[-1:] if segment_text.rstrip() else ''
            # 检查句子是否以句号结尾（。！？!?.）
            if last_char in '。！？!?.':
                pause_samples = int(args.sentence_pause * sr)
                silence = np.zeros(pause_samples, dtype=audio.dtype)
                audio = np.concatenate([audio, silence])
                logger.info(f"在片段末尾添加 {args.sentence_pause} 秒静音 (末尾字符: '{last_char}')")
            else:
                logger.debug(f"片段未以句号结尾，不添加静音 (末尾字符: '{last_char}')")
        
        all_audios.append(audio)
        all_segments.append(segment_text)
        segment_durations.append(len(audio) / sr)
        sample_rate = sr
    
    # 合并音频
    final_audio = np.concatenate(all_audios)
    total_duration = len(final_audio) / sample_rate
    
    # 保存音频
    sf.write(str(args.output), final_audio, sample_rate)
    
    elapsed = time.time() - start_time
    logger.info(f"音频已保存: {args.output}")
    logger.info(f"总时长: {total_duration:.2f} 秒")
    logger.info(f"处理耗时: {elapsed:.2f} 秒 (实时率: {total_duration/elapsed:.2f}x)")
    
    # 生成字幕
    if args.output_srt:
        srt_path = Path(args.output_srt)
        generate_srt(all_segments, segment_durations, srt_path, args.subtitle_max_chars, args.subtitle_min_chars)


if __name__ == "__main__":
    main()
