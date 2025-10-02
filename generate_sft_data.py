import shutil
import json
import os
import re
import h5py
import numpy as np
import imageio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import argparse
import logging
from utils import get_bin_search_keyframes, create_message_list, get_counting_keyframes, get_dusting_keyframes, decode_action_primitive, load_video_frames, align_frames_with_subsampling, get_high_level_instruction_counting, get_high_level_instruction_search, get_high_level_instruction_dusting

# default DROID setting
DEFAULT_HZ = 15

@dataclass
class SFTParams:
    file: str
    trajectory_name: str
    task: str
    wrist_camera_id: str
    third_person_camera_id: str
    resize_hw: Optional[Tuple[int, int]]
    output_dir: str
    frame_subsample: int
    recent_frames_length: int  
    keyframes_length: int 
    overwrite: bool
    include_img_text: bool
    include_only_text: bool
    prediction_horizon: int
    boundary_window: int

def create_single_sft_entry(
    frames: List[np.ndarray],
    actions: List[str],
    timestep: int,
    recent_frames_length: int,
    keyframes_length: int,
    frames_dir: str,
    frame_subsample: int,
    all_keyframe_indices: List[int],
    high_level_instructions: List[str],
    include_img_text: bool, 
    include_only_text: bool, 
    prediction_horizon: int, 
) -> Optional[Dict]:
    """
    Create a SFT entry for high-level primitive prediction and keyframe selection task.
    
    This function generates training data for a model to learn keyframe selection and primitive prediction.
    The task involves:
    1. Predicting the next primitive action to execute
    2. Identifying important keyframes within a recent context window
    
    The model receives:
    - Memory: Established keyframes (occurred before the current context), optionally with their primitives
    - Context: Recent frames showing the current situation
    - Instruction: High-level instruction describing the current task
    
    The model outputs:
    - Current primitive: The next action primitive to execute
    - Keyframe positions: Which frames in the context should be predicted as keyframes (1-indexed)
    - Keyframe primitives: The action primitives associated with predicted keyframes (optional)
    
    Args:
        frames: List of video frames as numpy arrays
        actions: List of action primitives corresponding to each frame
        timestep: Current timestep in the trajectory
        recent_frames_length: Number of recent frames to include in context
        keyframes_length: Number of keyframes to include in memory
        frames_dir: Directory where frame images are saved
        frame_subsample: Subsampling factor for frames (e.g., 3 means every 3rd frame)
        all_keyframe_indices: Indices of all keyframes in the trajectory
        high_level_instructions: High-level instructions at each timestep
        include_img_text: Whether to include both images and text in the output
        include_only_text: Whether to include only text (no images) in the output
        prediction_horizon: How many steps ahead to predict the primitive
        boundary_window: Window size in frames for boundary action labels
        
    Returns:
        Dictionary containing the QA entry with message list and metadata, or None if invalid
    """
    # calculate frame alignment based on subsampling
    # start from the correct offset to align with subsampled frames
    start_frame_offset = timestep % frame_subsample
    all_context_indices = list(range(start_frame_offset, timestep + 1, frame_subsample))
    
    # take only the most recent frames for context
    recent_frame_idx = all_context_indices[-recent_frames_length:]
    start_context_idx = recent_frame_idx[0] 
    
    # find keyframes that fall within the current context window, these are the candidate keyframes that the model should predict    
    candidate_keyframes = align_frames_with_subsampling(
        all_keyframe_indices, 
        start_context_idx,
        timestep + 1,
        frame_subsample,
    )
    
    # convert absolute keyframe indices to relative positions within context (1-indexed)
    # tells the model which positions in the context are keyframes
    relative_keyframe_idx_in_context = []
    for idx in candidate_keyframes:
        pos = recent_frame_idx.index(idx) + 1 
        relative_keyframe_idx_in_context.append(pos)
    
    # get the primitives associated with candidate keyframes
    candidate_keyframe_primitives = [actions[i] for i in candidate_keyframes]

    # get the keyframes that have been added to memory (occurred before the current context)
    memory_keyframes = align_frames_with_subsampling(
        all_keyframe_indices,
        start_frame_offset,
        start_context_idx,
        frame_subsample,
    )
    if keyframes_length <= 0:
        memory_keyframes = []
    else:
        memory_keyframes = memory_keyframes[-keyframes_length:] 
    
    # get the primitives for memory keyframes
    memory_primitives = [actions[i] for i in memory_keyframes]
    
    # file paths for memory keyframes
    keyframe_paths = [
        os.path.abspath(os.path.join(frames_dir, f"frame_{idx:03d}.png"))
        for idx in memory_keyframes if idx < len(frames)
    ]
    
    # file paths for recent frames (short-term video)
    short_term_video_paths = [
        os.path.abspath(os.path.join(frames_dir, f"frame_{idx:03d}.png"))
        for idx in recent_frame_idx if idx < len(frames)
    ]
    
    current_instruction = high_level_instructions[timestep] 
    
    # determine the target primitive to predict (with prediction horizon)
    # if prediction horizon extends beyond available actions, use current action
    current_primitive = actions[timestep+prediction_horizon*frame_subsample] if timestep+prediction_horizon*frame_subsample < len(actions) else actions[timestep]

    answer = {"current_primitive": current_primitive}
    
    # include keyframe positions if using memory
    if keyframes_length > 0:
        answer["keyframe_positions"] = relative_keyframe_idx_in_context if relative_keyframe_idx_in_context else []
    
    # include keyframe primitives if textual memory is requested
    if include_only_text or include_img_text:
        answer["keyframe_primitives"] = candidate_keyframe_primitives if relative_keyframe_idx_in_context else []

    assert len(keyframe_paths) == len(memory_primitives), f"Number of keyframes ({len(keyframe_paths)}) and memory primitives ({len(memory_primitives)}) do not match"

    # create the message list as input to the model (contains prompts, images, and expected answers)
    message_list = create_message_list(
        current_instruction=current_instruction,
        memory_length=keyframes_length,
        keyframe_paths=keyframe_paths,
        memory_primitives=memory_primitives,
        short_term_video_paths=short_term_video_paths,
        include_img_text=include_img_text,
        include_only_text=include_only_text,
        answer=answer
    )
    
    # single SFT entry with metadata
    return {
        "mode": "keyframe_selection",
        "message_list": message_list,
        "metadata": {
            "context_start": start_context_idx,
            "context_end": timestep + 1,
            "memory_keyframes_length": len(memory_keyframes),
            "candidate_keyframes_length": len(candidate_keyframes),
            "recent_frames_length": len(recent_frame_idx),
            "instruction": current_instruction,
            "answer": answer
        }
    }


def generate_sft_data(
    file: str,
    trajectory_name: str,
    task: str,
    wrist_camera_id: Optional[str],
    third_person_camera_id: Optional[str],
    resize_hw: Optional[Tuple[int, int]],
    output_dir: str,
    frame_subsample: int,
    recent_frames_length: int,
    keyframes_length: int,
    overwrite: bool,
    include_img_text: bool,
    include_only_text: bool,
    prediction_horizon: int,
    boundary_window: int,
) -> List[Dict]:
    """
    Generate SFT data for high-level primitive prediction and keyframe selection from a single HDF5 rollout.
    
    This function processes a single HDF5 file containing robot demonstration data and generates
    question-answer pairs for training a model to predict the next action primitive and select relevant keyframes.
    The function extracts video frames, identifies keyframes based on the task type,
    and creates training examples at each timestep.
    
    Args:
        file: Path to the HDF5 file containing the rollout data
        trajectory_name: Name identifier for this rollout (used for output directory naming)
        task: Task type ("counting", "bin_search", or "dusting") - determines keyframe detection strategy
        wrist_camera_id: Camera ID for wrist-mounted camera view
        third_person_camera_id: Camera ID for third-person camera view
        resize_hw: Optional tuple (height, width) to resize frames to
        output_dir: Base directory where generated data will be saved
        frame_subsample: Stride for subsampling frames (e.g., 5 means every 5th frame)
        recent_frames_length: Maximum number of recent frames to include in context
        keyframes_length: Maximum number of keyframes to include in memory
        overwrite: Whether to overwrite existing output files
        include_img_text: Whether to include both images and text labels for keyframes
        include_only_text: Whether to include only text labels (no images) for keyframes
        prediction_horizon: Number of frame_subsample steps ahead to predict primitive for
        boundary_window: Window size in frames for boundary action labels
        
    Returns:
        List of dictionaries, each containing a SFT entry with message list and metadata
    """
    rollout_dir = os.path.join(output_dir, trajectory_name)
    os.makedirs(rollout_dir, exist_ok=True)
    frames_dir = os.path.join(rollout_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # select appropriate high-level instruction extraction function based on task
    if task == "counting":
        get_high_level_instructions = get_high_level_instruction_counting
    elif task == "search":
        get_high_level_instructions = get_high_level_instruction_search
    elif task == "dusting":
        get_high_level_instructions = get_high_level_instruction_dusting

    qa_dir = os.path.join(rollout_dir, "qa")
    os.makedirs(qa_dir, exist_ok=True)

    with h5py.File(file, "r") as f:
        # load and concatenate wrist and third-person camera frames
        wrist_frames = load_video_frames(f, wrist_camera_id, resize_hw=resize_hw)
        third_person_frames = load_video_frames(f, third_person_camera_id, resize_hw=resize_hw)
        frames = [np.concatenate([wrist, third], axis=0) for wrist, third in zip(wrist_frames, third_person_frames)]

        high_level_instructions = get_high_level_instructions(f)
        actions = [decode_action_primitive(a) for a in f["observation"]["action_primitive_pressed"][()]]
        if task == "counting":
            # remove scoop numbering (first/second/third) to generalize primitive labels
            actions = [re.sub(r'place the (?:first|second|third) scoop of (.+) in the (.+) bowl', r'place a scoop of \1 in the \2 bowl', action) for action in actions]
    
    # save frames as PNG files
    for i, frame in tqdm(enumerate(frames), total=len(frames), desc="Saving frames"):
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        if overwrite or not os.path.exists(frame_path):
            imageio.imwrite(frame_path, frame)

    # detect keyframes (first or last frame of a subset of action primitives)
    if task == "dusting":
        all_keyframe_indices = get_dusting_keyframes(actions)
    elif task == "counting":
        all_keyframe_indices = get_counting_keyframes(actions)
    elif task == "search":
        all_keyframe_indices = get_bin_search_keyframes(actions)

    keyframes_dir = os.path.join(rollout_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)
    for idx in all_keyframe_indices:
        src_path = os.path.join(frames_dir, f"frame_{idx:03d}.png")
        if os.path.exists(src_path):
            dst_path = os.path.join(keyframes_dir, f"frame_{idx:03d}.png")
            shutil.copy2(src_path, dst_path)
        else:
            raise ValueError(f"Keyframe image file not found: {src_path}")
             
    # create concatenated primitives in a window before primitive boundaries to help with transitions
    if task == "dusting":
        actions_with_concat_labels = list(actions)
        window = boundary_window * frame_subsample  # window size in frames
        
        # find boundaries where primitive changes and create concatenated primitive for frames leading up to the boundary
        for boundary_idx in range(1, len(actions_with_concat_labels)):
            prev_action = actions_with_concat_labels[boundary_idx - 1]
            next_action = actions_with_concat_labels[boundary_idx]
            if next_action != prev_action:
                start = max(0, boundary_idx - window)
                concat_label = f"{prev_action} and {next_action}"
                for j in range(start, boundary_idx):
                    actions_with_concat_labels[j] = concat_label
        actions = actions_with_concat_labels

    qa_entries = []

    for timestep in range(0, len(frames)):
        # create a single SFT entry for each timestep in the trajectory
        qa_entry = create_single_sft_entry(
            frames=frames,
            actions=actions,
            timestep=timestep,
            recent_frames_length=recent_frames_length,
            keyframes_length=keyframes_length,
            frames_dir=frames_dir,
            frame_subsample=frame_subsample,
            all_keyframe_indices=all_keyframe_indices,
            high_level_instructions=high_level_instructions,
            include_img_text=include_img_text,
            include_only_text=include_only_text,
            prediction_horizon=prediction_horizon,
        )
        
        if qa_entry is not None:
            qa_entries.append(qa_entry)
            qa_path = os.path.join(qa_dir, f"qa_{timestep:03d}.json")
            if overwrite or not os.path.exists(qa_path):
                with open(qa_path, "w") as fp:
                    json.dump(qa_entry, fp, indent=4)
    
    logging.info(f"Generated {len(qa_entries)} SFT entries for {trajectory_name}")
    return qa_entries

def process_file_worker(params: SFTParams):
    return generate_sft_data(
            file=params.file,
            trajectory_name=params.trajectory_name,
            task=params.task,
            wrist_camera_id=params.wrist_camera_id,
            third_person_camera_id=params.third_person_camera_id,
            resize_hw=params.resize_hw,
            output_dir=params.output_dir,
            frame_subsample=params.frame_subsample,
            recent_frames_length=params.recent_frames_length,
            keyframes_length=params.keyframes_length,
            overwrite=params.overwrite,
            include_img_text=params.include_img_text,
            include_only_text=params.include_only_text,
            prediction_horizon=params.prediction_horizon,
            boundary_window=params.boundary_window,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SFT data from .h5 rollouts using DROID setup.")
    parser.add_argument("--h5_path", required=True, help="Directory containing .h5 files or a single .h5 file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", choices=["counting", "search", "dusting"], required=True)
    parser.add_argument("--frame_subsample", type=int, default=5, help="Subsample stride for frames")
    parser.add_argument("--recent_frames_length", type=int, default=8, help="Maximum number of recent frames")
    parser.add_argument("--keyframes_length", type=int, default=8, help="Maximum number of keyframes")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--wrist_camera_id", default="18650758_left", help="Camera ID for wrist camera")
    parser.add_argument("--third_person_camera_id", default="25916956_left", help="Camera ID for third person camera")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("H", "W"), default=[180, 320], help="Resize frames")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing SFT JSONs")
    parser.add_argument("--prediction_horizon", type=int, default=2, help="Number of frame_subsample steps ahead to predict primitive for")
    parser.add_argument("--boundary_window", type=int, default=3, help="Window size in frames for boundary action labels")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")  
    # adding text history (default is only images)
    parser.add_argument("--include_img_text", action="store_true", help="Include both the image and the label of the keyframes as memory")
    parser.add_argument("--include_only_text", action="store_true", help="Include only the primitive labels for the keyframes as memory, no images")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    if args.overwrite:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    h5_files = []
    for root, _, files in os.walk(args.h5_path):
        for file in sorted(files):
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))
    
    trajectory_name = []
    for file in h5_files:
        if os.path.basename(file) == "trajectory.h5":
            trajectory_name.append(os.path.basename(os.path.dirname(file)))
        else:
            trajectory_name.append(os.path.splitext(os.path.basename(file))[0])
    
    params_list = [
        SFTParams(
            file=file,
            trajectory_name=traj,
            task=args.task,
            wrist_camera_id=args.wrist_camera_id,
            third_person_camera_id=args.third_person_camera_id,
            resize_hw=tuple(args.resize) if args.resize else None,
            output_dir=args.output_dir,
            frame_subsample=args.frame_subsample,
            recent_frames_length=args.recent_frames_length,
            keyframes_length=args.keyframes_length,
            overwrite=args.overwrite,
            include_img_text=args.include_img_text,
            include_only_text=args.include_only_text,
            prediction_horizon=args.prediction_horizon,
            boundary_window=args.boundary_window,
        )
        for file, traj in zip(h5_files, trajectory_name)
    ]
    
    compiled_data = []
    qid_counter = 0
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for qa_entries in tqdm(executor.map(process_file_worker, params_list), total=len(params_list)):
            for qa in qa_entries:
                entry = {
                    "qid": str(qid_counter),
                    "mode": qa["mode"],
                    "message_list": qa["message_list"],
                    "metadata": qa["metadata"]
                }
                compiled_data.append(entry)
                qid_counter += 1
    
    compiled_path = os.path.join(args.output_dir, "compiled_data.json")
    with open(compiled_path, "w") as fp:
        json.dump(compiled_data, fp, indent=4)
    
    logging.info(f"Total compiled {len(compiled_data)} SFT entries to {compiled_path}")


if __name__ == "__main__":
    main()
