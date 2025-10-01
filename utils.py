from pathlib import Path
import re
import json
from typing import List, Optional, Dict, Any
import io
import imageio
import cv2
import numpy as np

BIN_SEARCH_SYSTEM_PROMPT = Path("/iris/u/jrpan/robo_memory/question_templates/bin_search_prompt.txt").read_text()

def decode_action_primitive(byte_arr):
    if isinstance(byte_arr, (bytes, bytearray)):
        return byte_arr.decode("utf-8", errors="ignore").replace("_", " ")
    return str(byte_arr)

def align_frames_with_subsampling(
    all_keyframe_indices: List[int],
    start_idx: int,
    end_idx: int, 
    frame_subsample: int,
) -> List[int]:
    """Align keyframes to subsample boundaries and remove duplicates while preserving order"""
    frame_indices = [((k - start_idx) // frame_subsample) * frame_subsample + start_idx 
                    for k in all_keyframe_indices if start_idx <= k < end_idx]
    return list(dict.fromkeys(frame_indices))

def load_video_frames(h5_file, camera_id, resize_hw = None, convert_to_rgb = True):
    """Extract frames (as numpy RGB images) from the mp4 bytes stored in the HDF5 file.

    Args:
        h5_file: opened h5py.File
        camera_id: key inside observation/videos mapping to mp4 bytes
        resize_hw: optional (height, width) to resize frames
        convert_to_rgb: whether to convert BGR to RGB because imageio reads video frames in BGR format by default   

    Returns:
        frames: np.ndarray of shape (T, H, W, 3) dtype uint8
    """
    if "observation" not in h5_file or "videos" not in h5_file["observation"]:
        raise KeyError("Could not find observation/videos group in this file.")

    videos_grp = h5_file["observation"]["videos"]
    if camera_id not in videos_grp:
        raise KeyError(f"Camera id {camera_id} not found in observation/videos group.")

    video_dataset = videos_grp[camera_id]
    video_bytes = video_dataset[()]
    if hasattr(video_bytes, 'tobytes'):
        video_bytes = video_bytes.tobytes()

    buf = io.BytesIO(video_bytes)
    reader = imageio.get_reader(buf, format="mp4")
    frames = []
    try:
        for frame in reader:
            if convert_to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_hw is not None:
                frame = cv2.resize(frame, (resize_hw[1], resize_hw[0]))
            frames.append(frame)
    finally:
        reader.close()

    return np.stack(frames, axis=0)

def parse_search_instruction(instruction: str) -> Dict[str, Optional[str]]:
    """
    Parse a bin search instruction to extract object and bin information.
    
    Handles two instruction formats that occurred during data collection:
    1. "take the [object] from the [source_bin] bin and place in the [target_bin] bin"
    2. "take the [object] from the [source_bin] bin and place it in the [target_bin] bin"
    
    Args:
        instruction: The instruction string to parse
        
    Returns:
        Dictionary containing:
            - 'object': The object to be moved (str or None if parsing failed)
            - 'source_bin': The source bin name (str or None if parsing failed)  
            - 'target_bin': The target bin name (str or None if parsing failed)
    """
    # define patterns for both instruction formats
    patterns = [
        r'take the (.+?) from the (.+?) bin and place in the (.+?) bin',
        r'take the (.+?) from the (.+?) bin and place it in the (.+?) bin'
    ]
    
    # try to match against each pattern
    for pattern in patterns:
        match = re.search(pattern, instruction)
        if match:
            return {
                'object': match.group(1).strip(),
                'source_bin': match.group(2).strip(),
                'target_bin': match.group(3).strip()
            }
    
    print(f"[ERROR] No match found for instruction: '{instruction}'")
    return {
        'object': None,
        'source_bin': None,
        'target_bin': None
    }
        

def get_high_level_instruction_search(h5_file):
    primitives_pressed = h5_file['observation']['action_primitive_pressed'][:]
    primitives_pressed = [p.decode('utf-8') for p in primitives_pressed]

    hi_instructs = []
    current_object = None
    
    for primitive in reversed(primitives_pressed):
        if primitive.startswith("take"):
            current_object = parse_search_instruction(primitive)['object']
        
        if current_object:
            hi_instructs.append(f"The robot's wrist and third-person camera feed is shown below. What primtive should the robot take to retrieve the {current_object} and put it in the white bin?")
        else:
            hi_instructs.append(None)
    
    return list(reversed(hi_instructs))

def get_high_level_instruction_dusting(h5_file):
    primitives_pressed = h5_file['observation']['action_primitive_pressed'][:]
    return ["What primtive should the robot take to remove the items from the shelves, dust the shelves, and place the items back on the shelves?" for _ in primitives_pressed]

def get_high_level_instruction_counting(h5_file):
    primitives_pressed = h5_file['observation']['action_primitive_pressed'][:]
    final_question = h5_file.attrs['high_level_instruction']
    return [final_question for _ in primitives_pressed]

def get_counting_keyframes(actions: List[str]) -> List[int]:
    """
    Get keyframe indices with counting-specific rules per action segment:
    - If action matches "place *": use the last index of the segment
    - Otherwise: do not append any keyframe for the segment
    """
    if not actions:
        return []

    place_re = re.compile(r"\bplace\b", re.IGNORECASE)

    keyframe_indices: List[int] = []

    # find the first non-None action
    i = 0
    while i < len(actions) and (actions[i] == "None" or actions[i] is None):
        i += 1

    while i < len(actions):
        action = actions[i]
        if action == "None" or action is None:
            i += 1
            continue

        j = i + 1
        while j < len(actions) and actions[j] == action:
            j += 1
        segment_end = j - 1

        action_text = action if isinstance(action, str) else str(action)

        if place_re.search(action_text):
            keyframe_indices.append(segment_end)
        else:
            pass

        i = j

    return keyframe_indices

def get_bin_search_keyframes(actions: List[str]) -> List[int]:
    """
    Get keyframe indices with bin search-specific rules per action segment:
    - If action starts with "look": use the last index of the segment
    - Otherwise: do not append any keyframe for the segment
    """
    if not actions:
        return []

    keyframe_indices: List[int] = []

    # find the first non-None action
    i = 0
    while i < len(actions) and (actions[i] == "None" or actions[i] is None):
        i += 1

    while i < len(actions):
        action = actions[i]
        if action == "None" or action is None:
            i += 1
            continue

        j = i + 1
        while j < len(actions) and actions[j] == action:
            j += 1
        segment_end = j - 1

        action_text = action if isinstance(action, str) else str(action)

        if action_text.lower().startswith("look"):
            keyframe_indices.append(segment_end)

        i = j

    return keyframe_indices

def get_dusting_keyframes(actions: List[str]) -> List[int]:
    """
    Get keyframe indices with dusting-specific rules per action segment:
    - If action matches "remove the object*" or "pick up duster": use the first index of the segment
    - If action matches "dust * shelf" or "place * on * shelf": use the last index of the segment
    - Otherwise: do not append any keyframe for the segment
    """
    if not actions:
        return []

    remove_object_re = re.compile(r"\bremove\s+the\s+object\b", re.IGNORECASE)
    pick_up_duster_re = re.compile(r"\bpick\b.*\bup\b.*\bduster\b", re.IGNORECASE)
    dust_shelf_re = re.compile(r"\bdust\b.*\bshelf\b", re.IGNORECASE)
    place_on_shelf_re = re.compile(r"\bplace\b.*\bon\b.*\bshelf\b", re.IGNORECASE)

    keyframe_indices: List[int] = []

    # find the first non-None action
    i = 0
    while i < len(actions) and (actions[i] == "None" or actions[i] is None):
        i += 1

    while i < len(actions):
        action = actions[i]
        if action == "None" or action is None:
            i += 1
            continue

        segment_start = i
        j = i + 1
        while j < len(actions) and actions[j] == action:
            j += 1
        segment_end = j - 1


        if remove_object_re.search(action) or pick_up_duster_re.search(action):
            keyframe_indices.append(segment_start)
        elif dust_shelf_re.search(action) or place_on_shelf_re.search(action):
            keyframe_indices.append(segment_end)

        i = j

    return keyframe_indices

def create_message_list(
    current_instruction: str,
    memory_length: int,
    keyframe_paths: List[str],
    memory_primitives: List[str],
    short_term_video_paths: List[str],
    include_img_text: bool = False,
    include_only_text: bool = False,
    answer: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Create a unified message list that formats data for model consumption.
    
    This function constructs a conversation format with system prompt and user content that includes:
    - Task instruction
    - Memory keyframes (either as images or text primitives)
    - Recent video frames
    - Optional action primitive labels for keyframes
    - Optional ground truth answer for training data
    
    Args:
        current_instruction: The current task instruction describing what the robot should accomplish
        memory_length: Number of keyframes to include in long-term memory (0 means no keyframes)
        keyframe_paths: List of file paths to keyframe images representing important moments from the full video
        memory_primitives: List of action primitive names corresponding to each keyframe
        short_term_video_paths: List of file paths to recent context frames showing robot's recent actions
        include_img_text: If True, include action primitive labels as text alongside keyframe images
        include_only_text: If True, include only text primitives without keyframe images
        answer: Optional ground truth answer dictionary to append as assistant response (used for training data)
    
    Returns:
        List of message dictionaries in the expected format
    """
    system_prompt = BIN_SEARCH_SYSTEM_PROMPT
    
    if memory_length > 0:
        system_prompt += "- keyframe_positions: list of frame positions (1-indexed) from the video input where actions change\n"
    
    if include_only_text or include_img_text:
        system_prompt += "- keyframe_primitives: list of action names for each keyframe\n"
    
    user_content = [{"text": f"Task: {current_instruction}\n"}]
    
    if include_only_text and memory_primitives:
        user_content[0]["text"] += "Here are the past action primitives that have been executed:\n[" + ", ".join(memory_primitives) + "]"
    elif keyframe_paths and memory_length > 0:
        user_content[0]["text"] += "Here are the selected frames from the entirety of the full video that are of particular importance:"
        for i, keyframe_path in enumerate(keyframe_paths):
            user_content.append({"image": keyframe_path})
            if (include_img_text and 
                memory_primitives and 
                i < len(memory_primitives)):
                user_content.append({"text": f"Action executed at this keyframe: {memory_primitives[i]}"})
    
    user_content.extend([
        {"text": "\nHere is a video of the most recent actions the robot has executed:"},
        {"video": short_term_video_paths}
    ])
    
    message_list = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user", "content": user_content}
    ]
    
    if answer is not None:
        message_list.append({"role": "assistant", "content": [{"text": json.dumps(answer)}]})
    
    return message_list 
