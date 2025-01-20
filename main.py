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
    ����YAMNetģ��
    """
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model

def load_class_names(yamnet_model):
    """
    ����YAMNet�������
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
        self.auto_gain = True  # �����Զ��������
        self.target_rms = 0.2  # Ŀ��RMS����
        self.current_gain = 1.0  # ��ʼ����
        self.silence_threshold = 0.01  # ������ֵ
        
        # ��ʼ��PyAudio
        self.p = pyaudio.PyAudio()
        
        # �г������豸
        print("\n���õ���Ƶ�����豸:")
        for i in range(self.p.get_device_count()):
            try:
                dev = self.p.get_device_info_by_index(i)
                if dev['maxInputChannels'] > 0:
                    print(f"�豸 {i}: {dev['name']}")
                    print(f"  ����ͨ��: {dev['maxInputChannels']}")
                    print(f"  Ĭ�ϲ�����: {int(dev['defaultSampleRate'])}")
            except Exception as e:
                continue
        
        # �豸ѡ��
        while True:
            try:
                device_index = int(input("\n��ѡ����Ƶ�����豸���: "))
                device_info = self.p.get_device_info_by_index(device_index)
                if device_info['maxInputChannels'] > 0:
                    break
            except:
                print("��Ч���豸��ţ�������")
        
        # ���Բ�ͬ������
        formats = [pyaudio.paInt16, pyaudio.paFloat32]
        rates = [int(device_info['defaultSampleRate']), 48000, 44100, 16000, 8000]
        
        for fmt in formats:
            for rate in rates:
                try:
                    print(f"\n����: ��ʽ={fmt}, ������={rate}Hz")
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
                    print("�ɹ�����Ƶ����")
                    print(f"ʹ���豸: {device_info['name']}")
                    print(f"��ʽ: {'Int16' if fmt == pyaudio.paInt16 else 'Float32'}")
                    print(f"������: {rate}Hz")
                    return
                except Exception as e:
                    print(f"����ʧ��: {str(e)}")
                    continue
        
        raise ValueError("δ���ҵ����õ���Ƶ����")

    def compute_rms(self, audio_data):
        """������ƵRMSֵ"""
        return np.sqrt(np.mean(np.square(audio_data)))

    def adjust_gain(self, audio_data, rms):
        """��̬��������"""
        if rms > 0:
            gain_factor = self.target_rms / rms
            # ƽ������仯
            self.current_gain = self.current_gain * 0.7 + gain_factor * 0.3
            # �������淶Χ
            self.current_gain = np.clip(self.current_gain, 0.1, 10.0)
        return audio_data * self.current_gain

    def process_audio_data(self, audio_data):
        """������Ƶ����"""
        # ת�����ݸ�ʽ
        if self.format == pyaudio.paInt16:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0
        else:
            audio_np = np.frombuffer(audio_data, dtype=np.float32)

        # ����RMS
        rms = self.compute_rms(audio_np)
        
        # ��ӡ����ָʾ��
        if rms > self.silence_threshold:
            bars = int(rms * 50)
            print(f"����: {'|' * bars} {rms:.3f}")
        
        # Ӧ���Զ�����
        if self.auto_gain and rms > self.silence_threshold:
            audio_np = self.adjust_gain(audio_np, rms)
        
        # �ز�����16kHz�������Ҫ��
        if self.sample_rate != 16000:
            audio_np = resample_audio(audio_np, self.sample_rate, 16000)
        
        return audio_np

    def start_streaming(self):
        """��ʼ��Ƶ��"""
        print("\n��ʼ¼��...")
        print("����������������������ʱ����ʾ��")
        
        while not self.stop_flag:
            try:
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_np = self.process_audio_data(audio_data)
                self.audio_queue.put(audio_np)
            except Exception as e:
                print(f"¼������: {str(e)}")
                break

    def stop_streaming(self):
        """ֹͣ��Ƶ��"""
        self.stop_flag = True
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()

def resample_audio(audio_data, from_rate, to_rate):
    """�ز�����Ƶ����"""
    # �����³���
    new_length = int(len(audio_data) * to_rate / from_rate)
    # ���Բ�ֵ�ز���
    x_old = np.linspace(0, 1, len(audio_data))
    x_new = np.linspace(0, 1, new_length)
    return np.interp(x_new, x_old, audio_data)

def process_audio(audio_streamer, yamnet_model, class_names):
    """������Ƶ��������ʵʱԤ��"""
    while not audio_streamer.stop_flag:
        if not audio_streamer.audio_queue.empty():
            try:
                audio_np = audio_streamer.audio_queue.get()
                
                # ʹ��YAMNet����Ԥ��
                scores, embeddings, spectrogram = yamnet_model(audio_np)
                
                # ��ȡǰ��������ܵ����
                top_3_indices = tf.argsort(scores[0], direction='DESCENDING')[:3]
                
                # ֻ�е���һ��Ԥ�����Ŷȴ�����ֵʱ�����
                top_confidence = scores[0][top_3_indices[0]]
                if top_confidence > 0.1:
                    print("\n��⵽������:")
                    for idx in top_3_indices:
                        class_name = class_names[idx]
                        confidence = scores[0][idx]
                        if confidence > 0.1:  # ֻ��ʾ���Ŷȴ���0.1�Ľ��
                            print(f"- {class_name} (���Ŷ�: {confidence:.2f})")
                
            except Exception as e:
                print(f"������Ƶʱ����: {str(e)}")

def main():
    try:
        print("����YAMNetģ��...")
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
        
        print("������60����Զ�ֹͣ...")
        time.sleep(60)
        print("\nֹͣ¼��...")
        audio_streamer.stop_streaming()
        
        audio_thread.join()
        process_thread.join()
        
    except KeyboardInterrupt:
        print("\n�û��жϣ�ֹͣ¼��...")
        if 'audio_streamer' in locals():
            audio_streamer.stop_streaming()
            if 'audio_thread' in locals():
                audio_thread.join()
            if 'process_thread' in locals():
                process_thread.join()
    except Exception as e:
        print(f"��������: {str(e)}")
    finally:
        print("�������")

if __name__ == "__main__":
    main()