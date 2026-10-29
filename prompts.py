# ============================================================
#  功能：按 16 类生成问答提示，并根据实际选项确定答案范围
#  适用场景：无人机第一人称视角城市导航视频问答
#
#  使用方式：
#      from prompts import build_prompt
#      prompt = build_prompt(qa_category, qa)
# ============================================================

import re

# ────────────────────────────────────────────────────────────
# 通用前缀（所有类别共用）
# ────────────────────────────────────────────────────────────
BASE_PREFIX = (
    "You are analyzing a video recorded from the first-person perspective of a drone "
    "navigating through an urban environment. "
    "The frames below are sampled in chronological order from the video — "
    "earlier frames appear first and represent earlier moments in time.\n\n"
)

# ────────────────────────────────────────────────────────────
# 通用后缀：由题目中的实际选项决定字母范围，不按类别猜测。
# ────────────────────────────────────────────────────────────
ANSWER_SUFFIX = (
    "\n\nAnswer format (strictly follow — no extra text allowed):\n"
    "Option: []\n"
    "Fill in exactly ONE letter from A to {last_option} inside the brackets.\n\n"
    "Confidence: []\n"
    "Rate your confidence in this answer on a scale of 1 to 5 (1 = very uncertain, 5 = very certain).\n"
    "Fill in exactly ONE number inside the brackets."
)

# ────────────────────────────────────────────────────────────
# 各类别专属提示词
# ────────────────────────────────────────────────────────────
CATEGORY_PROMPTS = {

    # ── 1. Action Generation ─────────────────────────────────
    # 最大类别（1183题）。给定导航目标，判断无人机的下一步动作。
    # 题库中此类别可有 A-G 七个选项，范围由题目文本决定。
    "Action Generation": (
        "CONTEXT: You are a drone navigating toward a specified destination. "
        "You have been given a navigation goal at the start of the video.\n\n"
        "YOUR TASK: Based on all the frames shown, determine the single best NEXT ACTION "
        "the drone should take to make progress toward the destination.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Identify your current position:\n"
        "           Look at the LAST few frames. Where are you now? "
        "           What is directly in front of you, to your left, to your right?\n"
        "  Step 2 — Locate the destination:\n"
        "           Where is the target relative to your current position and heading? "
        "           Is it above you, below you, ahead, behind, to the left, or to the right?\n"
        "  Step 3 — Determine the best single action:\n"
        "           Choose the ONE action that most directly reduces the distance or angular "
        "           offset to the destination:\n"
        "           • 'I rise'              — destination is above current altitude\n"
        "           • 'I descend'           — destination is below current altitude\n"
        "           • 'I fly forward'       — destination is roughly ahead, same altitude\n"
        "           • 'I turn left'         — destination is to the left of current heading\n"
        "           • 'I turn right'        — destination is to the right of current heading\n"
        "           • 'I rotate camera'     — need to reorient camera to see/track destination\n"
        "  Step 4 — Verify:\n"
        "           Would this action bring the drone measurably closer to or better aligned "
        "           with the destination? If not, reconsider.\n\n"
        "IMPORTANT: Focus on the most RECENT frames (current state), "
        "not on what happened in the middle of the video.\n"
    ),

    # ── 2. Landmark Position ─────────────────────────────────
    # 给定导航目标，判断从当前位置看目标的方向。
    "Landmark Position": (
        "CONTEXT: At the start of the video, the drone was given a navigation destination. "
        "The video shows the drone's journey toward that destination.\n\n"
        "YOUR TASK: Based on the drone's CURRENT position (shown in the last frames), "
        "determine the DIRECTION to the destination from where the drone is RIGHT NOW.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Understand the destination:\n"
        "           Read the question carefully. What is the exact target "
        "           (e.g., 'center of the tarmac', 'helipad', 'building entrance')?\n"
        "  Step 2 — Assess current position (from last frames):\n"
        "           What does the drone see right now? Which buildings, structures, "
        "           or landmarks are visible? Has the drone overshot, undershot, "
        "           or is it approaching from the correct side?\n"
        "  Step 3 — Determine relative direction:\n"
        "           Is the destination:\n"
        "           • Ahead / forward from current heading?\n"
        "           • Behind / the drone has passed it?\n"
        "           • To the north, south, east, or west?\n"
        "           • Above or below current altitude?\n"
        "           • Slightly offset to the left or right?\n"
        "  Step 4 — Match to the answer choices:\n"
        "           Select the option that most precisely describes the direction "
        "           from the drone's CURRENT POSITION to the DESTINATION.\n\n"
        "CRITICAL: The question asks about direction FROM the current location, "
        "not about whether the drone has arrived yet.\n"
    ),

    # ── 3. Progress Evaluation ───────────────────────────────
    # 给出完整导航指令，判断无人机完成了多少进度。
    "Progress Evaluation": (
        "CONTEXT: At the start of the video, the drone was given a detailed set of "
        "step-by-step navigation instructions. The video shows how far the drone "
        "has followed those instructions.\n\n"
        "YOUR TASK: Evaluate how much of the navigation instruction has been COMPLETED "
        "by the time of the last frame shown.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Decompose the navigation instructions:\n"
        "           Break the full instruction into individual sub-steps in order. "
        "           For example: [rise → turn right → fly forward → turn left → approach target]\n"
        "  Step 2 — Match completed steps to the video:\n"
        "           Go through the frames chronologically. Which sub-steps have already "
        "           been executed? Look for: altitude changes, turns, landmarks passed, "
        "           objects approached.\n"
        "  Step 3 — Identify the current step:\n"
        "           What step is the drone currently in the middle of? "
        "           What has it NOT done yet?\n"
        "  Step 4 — Estimate completion percentage:\n"
        "           How many of the total steps are done? "
        "           Is the drone at the beginning (0-25%), middle (25-75%), "
        "           or near the end (75-100%) of the task?\n"
        "  Step 5 — Select the matching answer:\n"
        "           Choose the option that best describes the stage the drone has reached.\n\n"
        "KEY INSIGHT: Pay close attention to the EXACT wording of the navigation instructions "
        "in the question — the answer depends on matching the instructions to what you observe.\n"
    ),

    # ── 4. Trajectory Captioning ─────────────────────────────
    # 描述无人机从起点到终点的完整路径。
    "Trajectory Captioning": (
        "CONTEXT: The video shows the drone's complete flight path from start to finish.\n\n"
        "YOUR TASK: Identify which answer choice BEST SUMMARIZES the drone's full movement route.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Note the starting position (first 2-3 frames):\n"
        "           Where does the journey begin? Describe the surroundings "
        "           (between buildings, above street, near rooftop, etc.)\n"
        "  Step 2 — Track all major movements in order:\n"
        "           List every significant action: ascend, descend, turn left, turn right, "
        "           fly straight, approach a building, pass a landmark.\n"
        "  Step 3 — Note the ending position (last 2-3 frames):\n"
        "           Where does the journey end? What is the final destination "
        "           (balcony, helipad, entrance, rooftop, etc.) and at what floor/altitude?\n"
        "  Step 4 — Reconstruct the overall path shape:\n"
        "           Was the path: straight, curved, zigzag, spiral, loop? "
        "           Was it mostly horizontal, mostly vertical, or mixed?\n"
        "  Step 5 — Match to the answer choices:\n"
        "           Identify which option correctly captures start location + "
        "           movement style + end location. Eliminate options with wrong "
        "           start points, wrong endings, or incorrect intermediate landmarks.\n\n"
        "WATCH OUT: Many wrong answers have the correct start but wrong end, or "
        "describe a reasonable path but not the one actually taken.\n"
    ),

    # ── 5. Goal Detection ────────────────────────────────────
    # 在最后一帧，目标是否可见？在视野中的哪里？
    "Goal Detection": (
        "CONTEXT: At the start of the video, the drone was assigned a navigation destination. "
        "The video shows the drone navigating toward that goal.\n\n"
        "YOUR TASK: At the LAST FRAME of the video, determine whether the destination "
        "is VISIBLE in the drone's view, and if so, WHERE it appears in the frame.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Identify the destination:\n"
        "           Read the question to know exactly what to look for "
        "           (helipad center, tarmac, balcony, building entrance, etc.)\n"
        "  Step 2 — Focus on the LAST frame:\n"
        "           This is the most important frame. Ignore the earlier journey.\n"
        "  Step 3 — Search the final frame systematically:\n"
        "           Scan upper-left, upper-center, upper-right, center-left, "
        "           center, center-right, lower-left, lower-center, lower-right.\n"
        "           Is the destination or any part of it visible anywhere?\n"
        "  Step 4 — Determine visibility:\n"
        "           • If visible: Report its screen position precisely "
        "             (e.g., 'lower right', 'upper center', 'directly ahead center')\n"
        "           • If not visible: The drone may not have arrived yet, "
        "             or may have passed the target\n"
        "  Step 5 — Select the matching answer:\n"
        "           Match your finding to the correct option, "
        "           paying attention to exact positional language.\n\n"
        "KEY RULE: Base your answer ONLY on the final frame. "
        "Do not infer from earlier frames where the destination 'should' be.\n"
    ),

    # ── 6. Cognitive Map ─────────────────────────────────────
    # 根据整个飞行过程构建空间认知地图，描述当前位置的周围环境。
    "Cognitive Map": (
        "CONTEXT: The drone has been navigating through an urban environment. "
        "You have seen the entire journey from start to the current location.\n\n"
        "YOUR TASK: Based on EVERYTHING you have observed throughout the video, "
        "describe what the SURROUNDING ENVIRONMENT looks like at the CURRENT LOCATION.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Build a spatial memory from all frames:\n"
        "           As you review the frames, mentally construct a map:\n"
        "           • What types of buildings are nearby (skyscrapers, mid-rise, residential)?\n"
        "           • What infrastructure is visible (roads, highways, bridges, parks)?\n"
        "           • What landmarks have been passed (domes, towers, helipads, plazas)?\n"
        "  Step 2 — Assess the current position (final frames):\n"
        "           What does the drone see RIGHT NOW in its immediate surroundings?\n"
        "           What is to the front, left, right, above, and below?\n"
        "  Step 3 — Integrate past observations with current view:\n"
        "           What structures that were seen earlier are now behind or around the drone? "
        "           Has the drone entered a new zone (residential vs commercial vs industrial)?\n"
        "  Step 4 — Characterize the environment:\n"
        "           Describe: building density, building heights, presence of open space, "
        "           road layout visible below, any green areas or water bodies.\n"
        "  Step 5 — Match to answer choices:\n"
        "           Select the option that best matches the environment around the drone's "
        "           CURRENT position based on your accumulated spatial memory.\n\n"
        "KEY INSIGHT: This question tests whether you have been tracking the full journey. "
        "The answer depends on combining what you see NOW with what you saw THROUGHOUT the video.\n"
    ),

    # ── 7. High-level Planning ───────────────────────────────
    # 从当前位置到目的地，下一步应该接近什么？
    "High-level Planning": (
        "CONTEXT: The drone has a navigation goal assigned at the start. "
        "The video shows progress made so far. The drone is at its current location.\n\n"
        "YOUR TASK: Determine what the drone should approach NEXT as its immediate "
        "intermediate waypoint to efficiently reach the final destination.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Understand the final destination:\n"
        "           Read the question carefully. What is the ultimate goal?\n"
        "  Step 2 — Assess current position from the last frames:\n"
        "           Where is the drone now? What surrounds it?\n"
        "  Step 3 — Estimate remaining distance and direction:\n"
        "           How far is the drone from the destination? "
        "           What is the general direction needed to reach it?\n"
        "  Step 4 — Identify the best NEXT intermediate landmark/waypoint:\n"
        "           What object, structure, or area should the drone head toward NEXT "
        "           to make progress? This should be:\n"
        "           • In the general direction of the destination\n"
        "           • A meaningful navigational stepping stone\n"
        "           • Not the final destination itself (this asks for the NEXT step)\n"
        "  Step 5 — Evaluate the answer choices:\n"
        "           Which option describes a waypoint that is: (a) consistent with the "
        "           drone's current position, (b) in the right direction, and "
        "           (c) a logical step toward the final goal?\n\n"
        "AVOID: Don't choose options that are behind the drone, already passed, "
        "or too far ahead skipping necessary intermediate steps.\n"
    ),

    # ── 8. Association Reasoning ─────────────────────────────
    # 目标是否在视野中？若不在，最相关的物体是什么及其位置？
    "Association Reasoning": (
        "CONTEXT: The drone was assigned a navigation goal at the start of the video. "
        "The video shows the drone navigating toward that goal.\n\n"
        "YOUR TASK (two-part question):\n"
        "  PART A — Is the destination currently visible in the frame?\n"
        "  PART B — If NOT visible: identify the most relevant visible object "
        "            and describe where it is in the frame.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Identify the navigation target:\n"
        "           What is the drone trying to reach (tarmac center, helipad, balcony, etc.)?\n"
        "  Step 2 — Examine the current frame carefully:\n"
        "           Scan the entire frame: top, center, bottom, left, right.\n"
        "           Is the destination or any recognizable part of it visible?\n"
        "  Step 3a — If the destination IS visible:\n"
        "            Report its location in the frame (upper/lower/left/right/center).\n"
        "  Step 3b — If the destination is NOT visible:\n"
        "            Identify the MOST RELEVANT nearby object — this means an object that:\n"
        "            • Is spatially closest to the destination based on the journey so far\n"
        "            • Could serve as a navigational reference point toward the goal\n"
        "            • Is prominently visible in the current frame\n"
        "  Step 4 — Report position of the relevant object:\n"
        "           Describe where in the frame this object appears "
        "           (upper-left, lower-right, center, etc.)\n"
        "  Step 5 — Match to the answer choices:\n"
        "           Select the option that correctly handles both parts of the question.\n\n"
        "KEY: This question tests spatial awareness + ability to identify navigational proxies "
        "when the direct target is out of view.\n"
    ),

    # ── 9. Start/End Position ────────────────────────────────
    # 识别起点和终点。
    "Start/End Position": (
        "CONTEXT: The video shows the drone's complete journey from its starting position "
        "to its ending position.\n\n"
        "YOUR TASK: Identify both WHERE the drone STARTS and WHERE it ENDS.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Examine the FIRST 2-3 frames carefully:\n"
        "           What does the starting environment look like?\n"
        "           • Is the drone between buildings, above a street, near a rooftop?\n"
        "           • What specific structures or features define this starting location?\n"
        "           • What floor level or altitude does it appear to be at?\n"
        "  Step 2 — Examine the LAST 2-3 frames carefully:\n"
        "           What does the ending environment look like?\n"
        "           • Where has the drone arrived? A balcony, rooftop, helipad, entrance?\n"
        "           • What floor, altitude, or landmark is the final destination?\n"
        "           • What surrounds the drone at the end?\n"
        "  Step 3 — Ignore the middle:\n"
        "           The intermediate journey is irrelevant for this question. "
        "           Do NOT let mid-journey locations confuse your answer.\n"
        "  Step 4 — Match the pair to the answer choices:\n"
        "           The correct answer must accurately describe BOTH the start AND the end. "
        "           Eliminate any option that gets either one wrong.\n\n"
        "COMMON TRAP: Some wrong answers swap start and end, or correctly describe one "
        "but use a different location for the other.\n"
    ),

    # ── 10. Counterfactual ───────────────────────────────────
    # 假设性推理：如果采用不同路径，能完成任务吗？
    "Counterfactual": (
        "CONTEXT: The video shows what the drone ACTUALLY DID to navigate to its destination.\n\n"
        "YOUR TASK: Reason about a HYPOTHETICAL ALTERNATIVE scenario described in the question. "
        "If the drone had taken a different action, could it still complete the task? "
        "How would the alternative route work?\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Understand what ACTUALLY happened:\n"
        "           Summarize the real path shown in the video: "
        "           start position, key maneuvers, end position.\n"
        "  Step 2 — Identify the hypothetical change:\n"
        "           What exactly is the question asking you to change? "
        "           (e.g., 'instead of flying past buildings, if you ascend immediately...')\n"
        "  Step 3 — Reason about the physical/spatial consequences:\n"
        "           If that different action were taken:\n"
        "           • Would the drone end up in a physically plausible position?\n"
        "           • Are there obstacles that would block the alternative path?\n"
        "           • Would the alternative route be longer, shorter, or impossible?\n"
        "           • Would it still reach the same destination?\n"
        "  Step 4 — Evaluate task completion feasibility:\n"
        "           Can the task still be completed via the alternative? "
        "           With more difficulty? With no difference? Or not at all?\n"
        "  Step 5 — Match to the answer choices:\n"
        "           Select the option that correctly assesses both: "
        "           (a) whether the task is completable, and "
        "           (b) a reasonable description of how the alternative would unfold.\n\n"
        "KEY: Ground your counterfactual in the actual geometry and layout "
        "observed in the video — do not reason in the abstract.\n"
    ),

    # ── 11. Duration ─────────────────────────────────────────
    # 比较两个动作或阶段所用时间的长短。
    "Duration": (
        "CONTEXT: The frames are UNIFORMLY SAMPLED from the video. "
        "This means equal numbers of frames = equal amounts of real time.\n\n"
        "YOUR TASK: Compare the TIME DURATION of two different actions or movement segments.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Identify the two segments being compared:\n"
        "           Read the question carefully. What are the two actions/phases to compare? "
        "           (e.g., 'descent to roof' vs 'approach to main entrance')\n"
        "  Step 2 — Locate each segment in the frame sequence:\n"
        "           • When does Segment A start and end (which frames)?\n"
        "           • When does Segment B start and end (which frames)?\n"
        "  Step 3 — Count frames for each segment:\n"
        "           Since sampling is uniform, more frames = longer duration.\n"
        "           Count (or estimate) how many frames belong to each segment.\n"
        "  Step 4 — Use visual cues to judge temporal length:\n"
        "           • Large visual changes across many frames = longer duration\n"
        "           • Small incremental changes across few frames = shorter duration\n"
        "           • Rapid changes within very few frames = brief action\n"
        "  Step 5 — Compare and select:\n"
        "           Which segment spans more frames? That one took longer.\n"
        "           If they span roughly the same number of frames, they are approximately equal.\n\n"
        "IMPORTANT: Do not rely on subjective impression of speed. "
        "Count the frames systematically.\n"
    ),

    # ── 12. Object Recall ────────────────────────────────────
    # 回忆某个位置或时刻，特定方向/位置的物体是什么。
    "Object Recall": (
        "CONTEXT: The drone navigated through an urban environment and observed many objects.\n\n"
        "YOUR TASK: Recall a SPECIFIC OBJECT at a SPECIFIC LOCATION or MOMENT in the video.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Understand the spatial/temporal reference in the question:\n"
        "           The question may ask about:\n"
        "           • A specific direction (to your right, above, below, behind)\n"
        "           • A specific moment (as you approach X, while passing Y)\n"
        "           • A specific location in the environment\n"
        "  Step 2 — Find the relevant frames:\n"
        "           Identify which frames correspond to the moment or location described. "
        "           Focus especially on those frames.\n"
        "  Step 3 — Examine the specified direction/area:\n"
        "           In those frames, carefully look at the area specified:\n"
        "           • 'To your right' = right side of the frame\n"
        "           • 'Directly above' = upper part of the frame\n"
        "           • 'Behind you' = may be visible in earlier frames as the drone passed it\n"
        "  Step 4 — Identify the object:\n"
        "           What specific object, structure, or feature appears in that location?\n"
        "           Note its type, size, shape, or color if helpful.\n"
        "  Step 5 — Match to the answer choices:\n"
        "           Select the option that correctly names the object at that "
        "           specific direction and moment.\n\n"
        "WATCH OUT: Some questions ask about objects that are only visible for 1-2 frames. "
        "Do not skip frames.\n"
    ),

    # ── 13. Proximity ────────────────────────────────────────
    # 无人机与特定物体之间的距离如何变化（增加/减少/不变）？
    "Proximity": (
        "CONTEXT: The drone performs a specific action while a particular object is visible.\n\n"
        "YOUR TASK: Determine how the DISTANCE between the drone and a specific object "
        "CHANGES as a result of the described action.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Identify the target object:\n"
        "           What specific object is the question asking about "
        "           (e.g., 'the cylindrical building with blue accents', "
        "           'the large glass building at the corner')?\n"
        "  Step 2 — Identify the action being analyzed:\n"
        "           What movement does the drone make? "
        "           (e.g., 'ascend slightly', 'turn slightly to the right', 'fly forward')\n"
        "  Step 3 — Find the relevant frames (before and after the action):\n"
        "           • Frames BEFORE the action: how large/close does the object appear?\n"
        "           • Frames AFTER the action: how large/close does the object appear now?\n"
        "  Step 4 — Judge the distance change using visual size as proxy:\n"
        "           • Object appears LARGER after action → drone got CLOSER (distance decreased)\n"
        "           • Object appears SMALLER after action → drone moved AWAY (distance increased)\n"
        "           • Object appears the SAME SIZE → distance is unchanged\n"
        "           Also consider: if the drone turns away, the object may disappear from view "
        "           (meaning the drone is no longer heading toward it, likely increasing distance).\n"
        "  Step 5 — Select the correct answer:\n"
        "           Choose: increases / decreases / remains the same.\n\n"
        "KEY TIP: Visual apparent size is the most reliable cue for distance. "
        "Larger = closer, smaller = farther.\n"
    ),

    # ── 14. Causal ───────────────────────────────────────────
    # 解释无人机执行某个特定动作的原因。
    "Causal": (
        "CONTEXT: The video shows the drone performing a specific action or maneuver "
        "as part of its navigation.\n\n"
        "YOUR TASK: Explain WHY the drone performed the specific action described in the question.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Identify the action to explain:\n"
        "           What specific maneuver is the question asking about? "
        "           (e.g., 'Why did you ascend to the height of the domed building?')\n"
        "  Step 2 — Locate this action in the video:\n"
        "           Find the frames where this specific maneuver occurs.\n"
        "  Step 3 — Examine frames BEFORE the action:\n"
        "           What was happening just before this action? "
        "           • Was there an obstacle ahead?\n"
        "           • Was the drone at the wrong altitude for its destination?\n"
        "           • Did something in the environment require this response?\n"
        "  Step 4 — Examine frames DURING and AFTER the action:\n"
        "           What does this maneuver achieve? "
        "           • Does it avoid something?\n"
        "           • Does it bring the drone closer to its goal?\n"
        "           • Does it align the drone with a specific target?\n"
        "  Step 5 — Identify the TRUE cause:\n"
        "           The cause should be: necessary (the drone had to do this), "
        "           and sufficient (this explains why this specific action and not another). "
        "           Distinguish from: coincidental timing, unrelated events, speculation.\n"
        "  Step 6 — Match to the answer choices:\n"
        "           Select the option that provides the most direct and accurate causal explanation.\n\n"
        "AVOID: Do not choose an answer just because it sounds plausible in isolation — "
        "it must be consistent with what is actually visible in the video.\n"
    ),

    # ── 15. Sequence Recall ──────────────────────────────────
    # 回忆某个事件之后（或之前）发生了什么动作。
    "Sequence Recall": (
        "CONTEXT: The frames are shown in strict chronological order.\n\n"
        "YOUR TASK: Recall the correct TEMPORAL ORDER of actions — specifically, "
        "what action came BEFORE or AFTER a given reference event.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Identify the reference event:\n"
        "           What is the anchor event in the question? "
        "           (e.g., 'after centering the conical structure in the frame', "
        "           'after hovering briefly')\n"
        "  Step 2 — Locate the reference event in the frame sequence:\n"
        "           Scan through the frames to find where this specific event occurs. "
        "           Which frame(s) show it happening?\n"
        "  Step 3 — Look at the NEXT frames (if asking 'what comes after'):\n"
        "           What does the drone do immediately following the reference event? "
        "           Describe the motion precisely: direction, altitude change, turn.\n"
        "  Step 4 — Look at PRECEDING frames (if asking 'what came before'):\n"
        "           What was the drone doing just before the reference event?\n"
        "  Step 5 — Eliminate wrong options:\n"
        "           Wrong answers may describe: actions from a different part of the video, "
        "           plausible-but-incorrect sequences, or actions in the wrong order.\n"
        "  Step 6 — Select the matching answer:\n"
        "           Choose the option that correctly describes the IMMEDIATELY "
        "           adjacent action in the correct temporal direction.\n\n"
        "KEY RULE: 'Next action' means the very FIRST thing that happens AFTER the event — "
        "not a later action, and not a simultaneous one.\n"
    ),

    # ── 16. Scene Recall ─────────────────────────────────────
    # 回忆在特定飞行阶段/动作期间观察到了哪些物体。
    "Scene Recall": (
        "CONTEXT: The drone flew through various parts of an urban environment.\n\n"
        "YOUR TASK: Recall what OBJECTS were observed during a SPECIFIC SEGMENT "
        "or ACTION described in the question.\n\n"
        "STEP-BY-STEP REASONING GUIDE:\n"
        "  Step 1 — Identify the specific segment being asked about:\n"
        "           What phase or action is the question referring to? "
        "           (e.g., 'during your descent along the side of the building', "
        "           'while flying between two tall buildings')\n"
        "  Step 2 — Isolate the relevant frames:\n"
        "           Find the start and end frames for this specific segment. "
        "           Focus ONLY on frames within this segment.\n"
        "  Step 3 — Catalog ALL objects visible in those frames:\n"
        "           Be systematic — scan each frame from top to bottom, left to right:\n"
        "           • Architectural elements (balconies, domes, facades, rooftops)\n"
        "           • Vegetation (trees, gardens, parks)\n"
        "           • Infrastructure (roads, bridges, lamps, signs)\n"
        "           • Other structures (towers, cylindrical buildings, archways)\n"
        "  Step 4 — Check against each answer option:\n"
        "           For each answer choice, verify: are ALL listed objects actually visible "
        "           in the segment? Is there any object listed that was NOT seen?\n"
        "  Step 5 — Select the most accurate set:\n"
        "           Choose the option whose object list is entirely correct — "
        "           no missing key objects, no objects that were not actually seen.\n\n"
        "WATCH OUT: Wrong answers often include one plausible but incorrect object. "
        "Verify EVERY item in the list before selecting.\n"
    ),
}

# ────────────────────────────────────────────────────────────
# 兜底提示（未收录类别时使用）
# ────────────────────────────────────────────────────────────
DEFAULT_HINT = (
    "Carefully analyze all frames in chronological order. "
    "Observe the drone's movements, the urban environment, and any objects or events. "
    "Use all available visual evidence to answer the question accurately.\n"
)

# ────────────────────────────────────────────────────────────
# 主函数：根据类别构建完整 Prompt
# ────────────────────────────────────────────────────────────
def detect_num_options(question: str) -> int:
    """读取实际选项标签；无选项的旧调用沿用默认五选一。"""
    letters = re.findall(
        r"^\s*(?:(?:choices?|choose|options?|answer)\s*[:：]\s*)*([A-G])[.)]\s",
        question, re.MULTILINE | re.IGNORECASE,
    )
    return max((ord(letter.upper()) - ord('A') + 1 for letter in letters), default=5)


def build_prompt(qa_category: str, question: str) -> str:
    """
    根据问题类别构建完整的 Prompt。

    Args:
        qa_category : 问题类别字符串，与数据集中 question_category 列一致
        question    : 原始问题内容（含题目实际选项，最高到 G）

    Returns:
        完整的 prompt 字符串，可直接用于 API 调用
    """
    category_hint = CATEGORY_PROMPTS.get(qa_category, DEFAULT_HINT)
    suffix = ANSWER_SUFFIX.format(last_option=chr(ord('A') + detect_num_options(question) - 1))

    prompt = (
        BASE_PREFIX
        + category_hint
        + "\nQuestion:\n"
        + question
        + suffix
    )
    return prompt


# ────────────────────────────────────────────────────────────
# 快速测试
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("/mnt/user-data/uploads/MCQ.parquet")
    all_cats = df["question_category"].unique()

    print(f"数据集中共 {len(all_cats)} 个类别")
    print(f"prompts.py 收录 {len(CATEGORY_PROMPTS)} 个类别\n")

    missing = [c for c in all_cats if c not in CATEGORY_PROMPTS]
    if missing:
        print(f"⚠️  未收录类别: {missing}")
    else:
        print("✅  全部类别已收录\n")

    # 打印每个类别的一个示例
    for cat in all_cats:
        sample_q = df[df["question_category"] == cat]["question"].iloc[0]
        print(f"{'='*60}")
        print(f"类别: {cat}")
        print(build_prompt(cat, sample_q))
        print()
