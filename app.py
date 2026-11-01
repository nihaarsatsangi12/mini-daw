"""
Mini DAW - Multi-Effect Vocal Processor.

Upload a recording, switch effects on with checkboxes, adjust them with
sliders, and download the processed file. Built with Gradio and Spotify's
Pedalboard, plus a few effects written by hand in dsp.py.

Run locally:  python app.py   then open http://localhost:7860
"""
import os
import tempfile
import uuid

import gradio as gr
import numpy as np
import soundfile as sf
from pedalboard import (
    Compressor,
    Delay,
    Distortion,
    HighpassFilter,
    LowpassFilter,
    NoiseGate,
    Pedalboard,
    PeakFilter,
    PitchShift,
    Reverb,
)

from dsp import (
    multi_pitch_octaves,
    normalize,
    run_board,
    schroeder_effect,
    simple_deesser,
    three_d_audio,
)

OUTPUT_DIR = tempfile.gettempdir()
MAX_SECONDS = 120  # keep processing quick on free hosting


def process_audio(
    audio_path,
    use_noise_gate, noise_thresh_db,
    use_highpass, hp_cutoff,
    use_lowpass, lp_cutoff,
    use_bandpass, bp_low_cut, bp_high_cut,
    use_eq, mud_cut_db, presence_gain_db, air_gain_db,
    use_comp, comp_thresh_db, comp_ratio,
    use_saturation, sat_drive_db,
    use_reverb_hall, hall_room_size, hall_wet,
    use_reverb_studio, studio_room_size, studio_wet,
    use_schroeder, schroeder_wet,
    use_delay, delay_seconds, delay_feedback, delay_mix,
    use_pitch_single, pitch_semitones,
    use_multi_pitch,
    use_deesser,
    use_3d_audio, rotation_speed, depth,
):
    if audio_path is None:
        raise gr.Error("Please upload an audio file first.")

    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)  # (samples, channels)
    if len(audio) > MAX_SECONDS * sr:
        raise gr.Error(f"Please upload a clip shorter than {MAX_SECONDS} seconds.")

    # ---- Corrective effects first: clean and shape the raw signal ----
    board = Pedalboard([])

    if use_noise_gate:
        board.append(NoiseGate(threshold_db=noise_thresh_db, ratio=4.0, attack_ms=5.0, release_ms=120.0))

    if use_highpass:
        board.append(HighpassFilter(cutoff_frequency_hz=hp_cutoff))
    if use_lowpass:
        board.append(LowpassFilter(cutoff_frequency_hz=lp_cutoff))
    if use_bandpass:
        board.append(HighpassFilter(cutoff_frequency_hz=bp_low_cut))
        board.append(LowpassFilter(cutoff_frequency_hz=bp_high_cut))

    if use_eq:
        board.append(PeakFilter(cutoff_frequency_hz=250.0, gain_db=mud_cut_db, q=1.0))        # mud
        board.append(PeakFilter(cutoff_frequency_hz=3500.0, gain_db=presence_gain_db, q=1.0))  # presence
        board.append(PeakFilter(cutoff_frequency_hz=10000.0, gain_db=air_gain_db, q=0.7))      # air

    if use_comp:
        board.append(Compressor(threshold_db=comp_thresh_db, ratio=comp_ratio, attack_ms=5.0, release_ms=120.0))

    # ---- Creative effects after: colour, space and time ----
    if use_saturation:
        board.append(Distortion(drive_db=sat_drive_db))

    if use_reverb_studio:
        board.append(Reverb(room_size=studio_room_size, damping=0.3,
                            wet_level=studio_wet, dry_level=1 - studio_wet, width=1.0))
    if use_reverb_hall:
        board.append(Reverb(room_size=hall_room_size, damping=0.4,
                            wet_level=hall_wet, dry_level=1 - hall_wet, width=1.0))

    if use_delay:
        board.append(Delay(delay_seconds=delay_seconds, feedback=delay_feedback, mix=delay_mix))

    if use_pitch_single:
        board.append(PitchShift(semitones=pitch_semitones))

    processed = run_board(board, audio, sr) if len(board) else audio

    # ---- Hand-written effects (dsp.py) ----
    if use_schroeder:
        processed = schroeder_effect(processed, sr, wet=schroeder_wet)
    if use_multi_pitch:
        processed = multi_pitch_octaves(processed, sr)
    if use_deesser:
        processed = simple_deesser(processed, sr)
    if use_3d_audio:
        processed = three_d_audio(processed, sr, rotation_speed, depth)

    processed = normalize(processed)

    out_path = os.path.join(OUTPUT_DIR, f"processed_{uuid.uuid4().hex[:8]}.wav")
    sf.write(out_path, processed, sr)
    return out_path, out_path


with gr.Blocks(title="Mini DAW") as demo:
    gr.Markdown(
        "# Mini DAW: Multi-Effect Vocal Processor\n"
        "Upload a vocal recording (WAV works best, under 2 minutes), tick the effects you want, "
        "adjust the sliders, then press **Process**."
    )

    with gr.Row():
        inp = gr.Audio(type="filepath", sources=["upload"], label="Upload Audio")
        out = gr.Audio(type="filepath", label="Processed Output")
        download = gr.File(label="Download Processed File")

    with gr.Row():
        # LEFT COLUMN: corrective effects
        with gr.Column():
            gr.Markdown("### 1. Noise Cleaning")
            use_noise_gate = gr.Checkbox(True, label="Enable Noise Gate")
            noise_thresh_db = gr.Slider(-80, 0, -50, label="Threshold (dB)")

            gr.Markdown("### 2. Filters")
            use_highpass = gr.Checkbox(True, label="High-pass Filter")
            hp_cutoff = gr.Slider(40, 200, 90, label="HP Cutoff (Hz)")
            use_lowpass = gr.Checkbox(False, label="Low-pass Filter")
            lp_cutoff = gr.Slider(1000, 8000, 6000, label="LP Cutoff (Hz)")
            use_bandpass = gr.Checkbox(False, label="Band-pass (HP + LP)")
            bp_low_cut = gr.Slider(200, 800, 300, label="BP Low Cut (Hz)")
            bp_high_cut = gr.Slider(2000, 6000, 3000, label="BP High Cut (Hz)")

            gr.Markdown("### 3. EQ (3-Band)")
            use_eq = gr.Checkbox(True, label="Enable EQ")
            mud_cut_db = gr.Slider(-12, 0, -4, label="Mud Cut (250 Hz, dB)")
            presence_gain_db = gr.Slider(0, 12, 4, label="Presence (3.5 kHz, dB)")
            air_gain_db = gr.Slider(0, 12, 3, label="Air (10 kHz, dB)")

            gr.Markdown("### 4. Compression")
            use_comp = gr.Checkbox(True, label="Enable Compressor")
            comp_thresh_db = gr.Slider(-40, 0, -22, label="Threshold (dB)")
            comp_ratio = gr.Slider(1, 6, 3, label="Ratio")

            gr.Markdown("### 5. Saturation")
            use_saturation = gr.Checkbox(False, label="Enable Saturation")
            sat_drive_db = gr.Slider(0, 24, 8, label="Drive (dB)")

        # RIGHT COLUMN: creative effects
        with gr.Column():
            gr.Markdown("### 6. Reverb")
            use_reverb_studio = gr.Checkbox(True, label="Studio Reverb (Small)")
            studio_room_size = gr.Slider(0.1, 1.0, 0.5, label="Studio Room Size")
            studio_wet = gr.Slider(0, 0.4, 0.12, label="Studio Wet Level")
            use_reverb_hall = gr.Checkbox(False, label="Hall Reverb (Large)")
            hall_room_size = gr.Slider(0.4, 1.0, 0.8, label="Hall Room Size")
            hall_wet = gr.Slider(0, 0.6, 0.25, label="Hall Wet Level")
            use_schroeder = gr.Checkbox(False, label="Schroeder Reverb (hand-written, 4 combs + 2 all-pass)")
            schroeder_wet = gr.Slider(0, 1, 0.3, label="Schroeder Wet Level")

            gr.Markdown("### 7. Delay")
            use_delay = gr.Checkbox(False, label="Enable Delay / Echo")
            delay_seconds = gr.Slider(0.05, 1.0, 0.35, label="Delay Time (s)")
            delay_feedback = gr.Slider(0, 0.9, 0.3, label="Feedback")
            delay_mix = gr.Slider(0, 1, 0.3, label="Mix")

            gr.Markdown("### 8. Pitch")
            use_pitch_single = gr.Checkbox(False, label="Enable Pitch Shift")
            pitch_semitones = gr.Slider(-12, 12, 0, step=1, label="Semitones")
            use_multi_pitch = gr.Checkbox(False, label="Multi-Pitch Harmonizer (0, ±12, +24 semitones)")

            gr.Markdown("### 9. De-Esser")
            use_deesser = gr.Checkbox(False, label="Enable De-Esser")

            gr.Markdown("### 10. 8D Audio (auto-pan)")
            use_3d_audio = gr.Checkbox(False, label="Enable 8D Rotation")
            rotation_speed = gr.Slider(0.1, 3.0, 0.5, label="Rotation Speed (Hz)")
            depth = gr.Slider(0.0, 1.0, 1.0, label="Depth / Width")

    run_btn = gr.Button("Process", variant="primary")
    run_btn.click(
        fn=process_audio,
        inputs=[
            inp,
            use_noise_gate, noise_thresh_db,
            use_highpass, hp_cutoff,
            use_lowpass, lp_cutoff,
            use_bandpass, bp_low_cut, bp_high_cut,
            use_eq, mud_cut_db, presence_gain_db, air_gain_db,
            use_comp, comp_thresh_db, comp_ratio,
            use_saturation, sat_drive_db,
            use_reverb_hall, hall_room_size, hall_wet,
            use_reverb_studio, studio_room_size, studio_wet,
            use_schroeder, schroeder_wet,
            use_delay, delay_seconds, delay_feedback, delay_mix,
            use_pitch_single, pitch_semitones,
            use_multi_pitch,
            use_deesser,
            use_3d_audio, rotation_speed, depth,
        ],
        outputs=[out, download],
    )


if __name__ == "__main__":
    # 0.0.0.0 and $PORT let the same file run locally and on hosting services
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
