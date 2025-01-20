# -*- coding: GBK -*-

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pyaudio
import wave
import csv
import queue
import threading
import time

def load_yamnet():
    """
    加载YAMNet模型
    """
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model

def load_class_names(yamnet_model):
    """
    加载YAMNet类别名称
    """
    class_map_path = yamnet_model.class_map_path().numpy()
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_names.append(row[2])
    return class_names

class AudioStreamer:
    def __init__(self, sample_rate=16000, chunk_size=16000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.stop_flag = False
        self.auto_gain = True  # 启用自动增益控制
        self.target_rms = 0.2  # 目标RMS音量
        self.current_gain = 1.0  # 初始增益
        self.silence_threshold = 0.01  # 静音阈值
        
        # 初始化PyAudio
        self.p = pyaudio.PyAudio()
        
        # 列出所有设备
        print("\n可用的音频输入设备:")
        for i in range(self.p.get_device_count()):
            try:
                dev = self.p.get_device_info_by_index(i)
                if dev['maxInputChannels'] > 0:
                    print(f"设备 {i}: {dev['name']}")
                    print(f"  输入通道: {dev['maxInputChannels']}")
                    print(f"  默认采样率: {int(dev['defaultSampleRate'])}")
            except Exception as e:
                continue
        
        # 设备选择
        while True:
            try:
                device_index = int(input("\n请选择音频输入设备编号: "))
                device_info = self.p.get_device_info_by_index(device_index)
                if device_info['maxInputChannels'] > 0:
                    break
            except:
                print("无效的设备编号，请重试")
        
        # 尝试不同的配置
        formats = [pyaudio.paInt16, pyaudio.paFloat32]
        rates = [int(device_info['defaultSampleRate']), 48000, 44100, 16000, 8000]
        
        for fmt in formats:
            for rate in rates:
                try:
                    print(f"\n尝试: 格式={fmt}, 采样率={rate}Hz")
                    self.stream = self.p.open(
                        format=fmt,
                        channels=1,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=min(2048, rate // 8)
                    )
                    self.format = fmt
                    self.sample_rate = rate
                    print("成功打开音频流！")
                    print(f"使用设备: {device_info['name']}")
                    print(f"格式: {'Int16' if fmt == pyaudio.paInt16 else 'Float32'}")
                    print(f"采样率: {rate}Hz")
                    return
                except Exception as e:
                    print(f"配置失败: {str(e)}")
                    continue
        
        raise ValueError("未能找到可用的音频配置")

    def compute_rms(self, audio_data):
        """计算音频RMS值"""
        return np.sqrt(np.mean(np.square(audio_data)))

    def adjust_gain(self, audio_data, rms):
        """动态调整增益"""
        if rms > 0:
            gain_factor = self.target_rms / rms
            # 平滑增益变化
            self.current_gain = self.current_gain * 0.7 + gain_factor * 0.3
            # 限制增益范围
            self.current_gain = np.clip(self.current_gain, 0.1, 10.0)
        return audio_data * self.current_gain

    def process_audio_data(self, audio_data):
        """处理音频数据"""
        # 转换数据格式
        if self.format == pyaudio.paInt16:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0
        else:
            audio_np = np.frombuffer(audio_data, dtype=np.float32)

        # 计算RMS
        rms = self.compute_rms(audio_np)
        
        # 打印音量指示器
        if rms > self.silence_threshold:
            bars = int(rms * 50)
            print(f"音量: {'|' * bars} {rms:.3f}")
        
        # 应用自动增益
        if self.auto_gain and rms > self.silence_threshold:
            audio_np = self.adjust_gain(audio_np, rms)
        
        # 重采样到16kHz（如果需要）
        if self.sample_rate != 16000:
            audio_np = resample_audio(audio_np, self.sample_rate, 16000)
        
        return audio_np

    def start_streaming(self):
        """开始音频流"""
        print("\n开始录音...")
        print("音量监视器已启动（静音时不显示）")
        
        while not self.stop_flag:
            try:
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_np = self.process_audio_data(audio_data)
                self.audio_queue.put(audio_np)
            except Exception as e:
                print(f"录音出错: {str(e)}")
                break

    def stop_streaming(self):
        """停止音频流"""
        self.stop_flag = True
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()

def resample_audio(audio_data, from_rate, to_rate):
    """重采样音频数据"""
    # 计算新长度
    new_length = int(len(audio_data) * to_rate / from_rate)
    # 线性插值重采样
    x_old = np.linspace(0, 1, len(audio_data))
    x_new = np.linspace(0, 1, new_length)
    return np.interp(x_new, x_old, audio_data)

def process_audio(audio_streamer, yamnet_model, class_names):
    """处理音频流并进行实时预测"""
    while not audio_streamer.stop_flag:
        if not audio_streamer.audio_queue.empty():
            try:
                audio_np = audio_streamer.audio_queue.get()
                
                # 使用YAMNet进行预测
                scores, embeddings, spectrogram = yamnet_model(audio_np)
                
                # 获取前三个最可能的类别
                top_3_indices = tf.argsort(scores[0], direction='DESCENDING')[:3]
                
                # 只有当第一个预测置信度大于阈值时才输出
                top_confidence = scores[0][top_3_indices[0]]
                if top_confidence > 0.1:
                    print("\n检测到的声音:")
                    for idx in top_3_indices:
                        class_name = class_names[idx]
                        confidence = scores[0][idx]
                        if confidence > 0.1:  # 只显示置信度大于0.1的结果
                            print(f"- {class_name} (置信度: {confidence:.2f})")
                
            except Exception as e:
                print(f"处理音频时出错: {str(e)}")

def main():
    try:
        print("加载YAMNet模型...")
        yamnet_model = load_yamnet()
        class_names = load_class_names(yamnet_model)
        
        audio_streamer = AudioStreamer()
        
        audio_thread = threading.Thread(
            target=audio_streamer.start_streaming
        )
        process_thread = threading.Thread(
            target=process_audio,
            args=(audio_streamer, yamnet_model, class_names)
        )
        
        audio_thread.start()
        process_thread.start()
        
        print("程序将在60秒后自动停止...")
        time.sleep(60)
        print("\n停止录音...")
        audio_streamer.stop_streaming()
        
        audio_thread.join()
        process_thread.join()
        
    except KeyboardInterrupt:
        print("\n用户中断，停止录音...")
        if 'audio_streamer' in locals():
            audio_streamer.stop_streaming()
            if 'audio_thread' in locals():
                audio_thread.join()
            if 'process_thread' in locals():
                process_thread.join()
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        print("程序结束")

if __name__ == "__main__":
    main()