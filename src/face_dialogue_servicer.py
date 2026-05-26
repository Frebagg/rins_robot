#!/usr/bin/env python3

import os
import json
import re
import tempfile
import time
import wave
from collections import defaultdict

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - runtime dependency guard
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - runtime dependency guard
    WhisperModel = None

from rins_robot.srv import FaceDialogue, Speech


class FaceDialogueServicer(Node):

    def __init__(self):
        super().__init__("face_dialogue_servicer")

        self.callback_group = ReentrantCallbackGroup()

        self.greet_client = self.create_client(
            Speech,
            "/greet_service",
            callback_group=self.callback_group,
        )
        self.dialogue_server = self.create_service(
            FaceDialogue,
            "/face_dialogue_service",
            self.handle_dialogue,
            callback_group=self.callback_group,
        )
        self.task_history_publisher = self.create_publisher(
            String,
            "/face_dialogue_task_history",
            10,
        )

        self.sample_rate = 16000
        self.utterance_seconds = 5.0
        self.max_turns = 8
        self.match_threshold = 0.45
        self.confirm_words = {
            "yes",
            "yeah",
            "yep"
        }

        self.task_history = []
        self.next_task_id = 1

        self.whisper_model = self._load_whisper_model()

        self.get_logger().info("Face dialogue service node initialized!")

    def _load_whisper_model(self):
        if WhisperModel is None:
            self.get_logger().error("faster-whisper is not installed.")
            return None

        model_name = os.environ.get("RINS_WHISPER_MODEL", "base.en")
        preferred_device = os.environ.get(
            "RINS_WHISPER_DEVICE",
            "cuda" if self._cuda_available() else "cpu"
        ).strip().lower()
        requested_compute_type = os.environ.get("RINS_WHISPER_COMPUTE_TYPE")

        if requested_compute_type:
            attempts = [(preferred_device, requested_compute_type.strip().lower())]
        elif preferred_device == "cuda":
            attempts = [
                ("cuda", "float16"),
                ("cuda", "float32"),
                ("cpu", "int8"),
            ]
        else:
            attempts = [
                ("cpu", "int8"),
                ("cpu", "float32"),
            ]

        last_error = None
        for device, compute_type in attempts:
            try:
                self.get_logger().info(f"Loading WhisperModel '{model_name}' on {device} ({compute_type})")
                return WhisperModel(model_name, device=device, compute_type=compute_type)
            except Exception as exc:
                last_error = exc
                self.get_logger().warn(
                    f"Could not load WhisperModel on {device} ({compute_type}): {exc}"
                )

        self.get_logger().error(f"Could not load WhisperModel after fallback attempts: {last_error}")
        return None

    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def speak(self, text):
        if not self.greet_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/greet_service is not available")
            return False

        request = Speech.Request()
        request.data = text
        future = self.greet_client.call_async(request)
        while rclpy.ok() and not future.done():
            time.sleep(0.01)

        response = future.result() if future.done() else None
        return response is not None and response.success

    def say(self, text):
        success = self.speak(text)
        time.sleep(0.7)
        return success

    def record_audio(self):
        if sd is None:
            self.get_logger().error("sounddevice is not installed.")
            return None

        frames = int(self.sample_rate * self.utterance_seconds)
        try:
            audio = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            return np.squeeze(audio)
        except Exception as exc:
            self.get_logger().error(f"Could not record audio: {exc}")
            return None

    def save_audio(self, audio):
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, "face_dialogue_utterance.wav")
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767).astype(np.int16)

        with wave.open(file_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm16.tobytes())

        return file_path

    def transcribe(self, audio_path):
        if self.whisper_model is None:
            return ""

        segments, _info = self.whisper_model.transcribe(
            audio_path,
            language="en",
            vad_filter=True,
            beam_size=3,
            initial_prompt="""
            Possible commands are:
            inspect the barrels, inspect barrels
            count rings, count the rings,
            detect anomalies in the red cell, detect the anomalies in the red cell,
            detect anomalies in the green cell, detect the anomalies in the green cell,
            nothing,
            yes.
            """
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()
        self.get_logger().info(f"Transcribed: '{text}'")
        return text

    def normalize_text(self, text):
        normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def parse_task(self, text):
        normalized = self.normalize_text(text)
        words = set(normalized.split())

        if not words:
            return None

        if "nothing" in words:
            return {"task": "nothing", "cell": "none"}

        if "barrel" in words or "barrels" in words:
            return {"task": "barrels", "cell": "none"}

        if "ring" in words or "rings" in words:
            return {"task": "rings", "cell": "none"}

        if "anomaly" in words or "anomalies" in words:
            if "red" in words:
                return {"task": "anomaly", "cell": "red"}
            if "green" in words:
                return {"task": "anomaly", "cell": "green"}

        return None

    def prompt_text(self, gender):
        if gender == "F":
            return "Hi woman, which task should I perform?"
        return "Hi man, which task should I perform?"

    def confirmation_text(self, selection):
        task = selection["task"]
        cell = selection["cell"]

        if task == "barrels":
            return "OK. I will inspect the barrels. Are you sure?"
        if task == "rings":
            return "OK. I will count the rings. Are you sure?"
        if task == "anomaly" and cell == "red":
            return "OK. I will detect anomalies in the red cell. Are you sure?"
        if task == "anomaly" and cell == "green":
            return "OK. I will detect anomalies in the green cell. Are you sure?"
        return "OK. I won't do anything. Are you sure?"

    def final_text(self, selection):
        task = selection["task"]
        cell = selection["cell"]

        if task == "barrels":
            return "OK. I will inspect the barrels"
        if task == "rings":
            return "OK. I will count the rings"
        if task == "anomaly" and cell == "red":
            return "OK. I will detect the anomalies in the red cell"
        if task == "anomaly" and cell == "green":
            return "OK. I will detect the anomalies in the green cell"
        return "OK. I wont do anything"

    def record_task_history(self, name, task, cell):
        entry = {
            "id": self.next_task_id,
            "name": name,
            "task": task,
            "cell": cell,
        }
        self.next_task_id += 1
        self.task_history.append(entry)

        message = String()
        message.data = json.dumps(self.task_history)
        self.task_history_publisher.publish(message)
        self.get_logger().info(f"Published task history entry {entry['id']} for name={name}, task={task}, cell={cell}")

    def finish_dialogue(self, res, name, selection):
        res.task = selection["task"]
        res.cell = selection["cell"]
        res.success = True
        self.record_task_history(name, selection["task"], selection["cell"])
        return res

    def listen_once(self):
        audio = self.record_audio()
        if audio is None:
            return ""

        audio_path = self.save_audio(audio)
        try:
            return self.transcribe(audio_path)
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    def ask_and_listen(self, phrase):
        self.speak(phrase)
        return self.listen_once()

    def handle_dialogue(self, req, res):
        name = (req.name or "").strip() or "UNKNOWN"
        gender = (req.gender or "M").strip().upper()
        if gender not in {"M", "F"}:
            gender = "M"

        self.say(self.prompt_text(gender))

        if gender == "M":
            first_reply = self.listen_once()
            selection = self.parse_task(first_reply) or {"task": "nothing", "cell": "none"}
            response_text = self.final_text(selection)
            self.say(response_text)
            return self.finish_dialogue(res, name, selection)

        task_counts = defaultdict(int)
        current_selection = None
        turns = 0

        # First female reply decides the initial task.
        while turns < self.max_turns:
            reply = self.listen_once()
            turns += 1

            selection = self.parse_task(reply)
            if selection is None:
                self.say(self.prompt_text(gender))
                continue

            current_selection = selection
            task_counts[(selection["task"], selection["cell"])] += 1

            if task_counts[(selection["task"], selection["cell"])] >= 2:
                final_response = self.final_text(selection)
                self.say(final_response)
                return self.finish_dialogue(res, name, selection)

            self.say(self.confirmation_text(selection))

            while turns < self.max_turns:
                reply = self.listen_once()
                turns += 1
                normalized = self.normalize_text(reply)
                words = set(normalized.split())

                if any(word in words for word in self.confirm_words):
                    final_response = self.final_text(current_selection)
                    self.say(final_response)
                    return self.finish_dialogue(res, name, current_selection)

                selection = self.parse_task(reply)
                if selection is None:
                    self.say(self.confirmation_text(current_selection))
                    continue

                current_selection = selection
                task_counts[(selection["task"], selection["cell"])] += 1

                if task_counts[(selection["task"], selection["cell"])] >= 2:
                    final_response = self.final_text(selection)
                    self.say(final_response)
                    return self.finish_dialogue(res, name, selection)

                self.say(self.confirmation_text(selection))

        if current_selection is None:
            current_selection = {"task": "nothing", "cell": "none"}

        final_response = self.final_text(current_selection)
        self.say(final_response)
        return self.finish_dialogue(res, name, current_selection)


def main(args=None):
    rclpy.init(args=args)
    node = FaceDialogueServicer()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()