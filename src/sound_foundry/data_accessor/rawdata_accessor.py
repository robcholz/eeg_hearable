from pathlib import Path
from typing import Optional, List, Mapping, Dict, Tuple

from sound_foundry.config import get_raw_dataset_path
from sound_foundry.data_accessor import download_data

# do not edit this
_ORIGINAL_MAP: Mapping[str, List[str]] = {
    # -------- animals --------
    "animal": [
        "Animal",
        "Domestic_animals_and_pets",
        "Livestock_and_farm_animals_and_working_animals",
        "Wild_animals",
        "Dog",
        "dog",
        "Cat",
        "cat",
        "Bird",
        "Bird_vocalization_and_bird_call_and_bird_song",
        "chirping_birds",
        "Chicken_and_rooster",
        "hen",
        "rooster",
        "Fowl",
        "Frog",
        "frog",
        "Cricket",
        "crickets",
        "Insect",
        "insects",
        "Crow",
        "crow",
        "Gull_and_seagull",
        "Buzz",
        "Meow",
        "Purr",
        "Bark",
        "Growling",
        "cow",
        "pig",
        "sheep",
    ],
    # -------- human voice & group --------
    "human_voice": [
        "Human_voice",
        "Speech",
        "Conversation",
        "Chatter",
        "Male_speech_and_man_speaking",
        "Female_speech_and_woman_speaking",
        "Child_speech_and_kid_speaking",
        "Speech_synthesizer",
        "Whispering",
        "Shout",
        "Yell",
        "Screaming",
        "Screech",
        "Laughter",
        "laughing",
        "Giggle",
        "Chuckle_and_chortle",
        "Cheering",
        "Applause",
        "Clapping",
        "clapping",
        "Crowd",
        "Human_group_actions",
        "Crying_and_sobbing",
        "crying_baby",
        "baby",
    ],
    # -------- body / close-range human actions --------
    "body_sounds": [
        "Hands",
        "Finger_snapping",
        "Breathing",
        "breathing",
        "Respiratory_sounds",
        "Cough",
        "coughing",
        "Sneeze",
        "sneezing",
        "Burping_and_eructation",
        "Fart",
        "Chewing_and_mastication",
        "drinking_sipping",
        "Gasp",
        "Sigh",
        "snoring",
        "Walk_and_footsteps",
        "footsteps",
        "Run",
        "brushing_teeth",
        "Scratching_(performance_technique)",
    ],
    # -------- music --------
    "music": [
        "Music",
        "Singing",
        "Male_singing",
        "Female_singing",
        "Musical_instrument",
        "Guitar",
        "Acoustic_guitar",
        "Electric_guitar",
        "Bass_guitar",
        "Piano",
        "Organ",
        "Accordion",
        "Harp",
        "Harmonica",
        "Keyboard_(musical)",
        "Trumpet",
        "Brass_instrument",
        "Bowed_string_instrument",
        "Plucked_string_instrument",
        "Wind_instrument_and_woodwind_instrument",
        "Strum",
    ],
    "percussion": [
        "Percussion",
        "Drum",
        "Drum_kit",
        "audio-drum",
        "Bass_drum",
        "Snare_drum",
        "Crash_cymbal",
        "Cymbal",
        "Hi-hat",
        "Tambourine",
        "Cowbell",
        "Gong",
        "Tabla",
        "Mallet_percussion",
        "Marimba_and_xylophone",
        "Glockenspiel",
        "Rattle_(instrument)",
    ],
    # -------- vehicles --------
    "aircraft": [
        "Aircraft",
        "Fixed-wing_aircraft_and_airplane",
        "airplane",
        "helicopter",
    ],
    "vehicle_ground": [
        "Vehicle",
        "Motor_vehicle_(road)",
        "Car",
        "Bus",
        "Truck",
        "Motorcycle",
        "Race_car_and_auto_racing",
        "Car_passing_by",
        "Bicycle",
        "Skateboard",
        "Train",
        "train",
        "Rail_transport",
        "Subway_and_metro_and_underground",
    ],
    "vehicle_water": [
        "Boat_and_Water_vehicle",
    ],
    "vehicle_sounds": [
        "Engine",
        "engine",
        "Engine_starting",
        "Idling",
        "Accelerating_and_revving_and_vroom",
        "Vehicle_horn_and_car_horn_and_honking",
        "car_horn",
        "Traffic_noise_and_roadway_noise",
        "Whoosh_and_swoosh_and_swish",
    ],
    # -------- water --------
    "water": [
        "Water",
        "water",
        "Liquid",
        "Stream",
        "Ocean",
        "Waves_and_surf",
        "sea_waves",
        "Drip",
        "Trickle_and_dribble",
        "Pour",
        "pouring_water",
        "Fill_(with_liquid)",
        "Bathtub_(filling_or_washing)",
        "Sink_(filling_or_washing)",
        "Water_tap_and_faucet",
        "Splash_and_splatter",
        "Boiling",
        "Gurgling",
        "Toilet_flush",
        "toilet_flush",
        "can_opening",
    ],
    # -------- weather / nature --------
    "weather": [
        "Rain",
        "rain",
        "Raindrop",
        "water_drops",
        "Wind",
        "wind",
        "Thunder",
        "Thunderstorm",
        "thunderstorm",
        "Fire",
        "crackling_fire",
        "Fireworks",
        "fireworks",
    ],
    # -------- domestic / tools / objects --------
    "domestic_and_objects": [
        "Domestic_sounds_and_home_sounds",
        "Door",
        "Sliding_door",
        "Doorbell",
        "Knock",
        "door_wood_knock",
        "door_wood_creaks",
        "Drawer_open_or_close",
        "Cupboard_open_or_close",
        "Dishes_and_pots_and_pans",
        "Cutlery_and_silverware",
        "Packing_tape_and_duct_tape",
        "Scissors",
        "Zipper_(clothing)",
        "Wood",
        "Writing",
        "Typewriter",
        "Typing",
        "Computer_keyboard",
        "keyboard_typing",
        "Printer",
        "printer",
        "mouse_click",
        "Keys_jangling",
        "Coin_(dropping)",
        "Glass",
        "Shatter",
        "glass_breaking",
        "Crack",
        "Crackle",
        "Crumpling_and_crinkling",
        "Crushing",
        "Tap",
        "Tearing",
        "Squeak",
        "Thump_and_thud",
        "Chink_and_clink",
        "Camera",
        "Chirp_and_tweet",
        "Hiss",
        "Ratchet_and_pawl",
        "Rattle",
    ],
    "tools_and_machines": [
        "Tools",
        "Power_tool",
        "Drill",
        "Hammer",
        "Sawing",
        "hand_saw",
        "chainsaw",
        "Mechanisms",
        "Mechanical_fan",
        "fan",
        "Microwave_oven",
        "blender",
        "dishwasher",
        "washing_machine",
        "vacuum_cleaner",
        "electric_shaver_toothbrush",
        "Frying_(food)",
        "frying",
    ],
    # -------- alerts / signals --------
    "alerts": [
        "Alarm",
        "clock_alarm",
        "Siren",
        "siren",
        "Ringtone",
        "Telephone",
        "Bell",
        "Church_bell",
        "church_bells",
        "Chime",
        "Wind_chime",
        "Bicycle_bell",
        "Clock",
        "Tick",
        "Tick-tock",
        "clock_tick",
    ],
    # -------- impacts --------
    "impacts": [
        "Boom",
        "Explosion",
        "Gunshot_and_gunfire",
        "Slam",
    ],
    # -------- dataset meta --------
    "audio_meta": [
        "audio-bass",
        "audio-mixture",
        "audio-rest",
        "audio-vocal",
    ],
}

# do not edit this
_OPTIMIZED_MAP: Mapping[str, List[Tuple[str, str]]] = {}


# this should be called in the global so it will only run once in initialization
# we use `category` to describe the labels for data from download_data.*, they are dirty and have some duplicated semantic meanings,
# we use `label` to describe the labels that are unique and have no semantic conflicts.
# 1. load original map from unified label to array of categories (just map[str,list[str]])
# 2. load all the categories(labels) from download_data.get_audio_categories(dataset_path,dataset=None)
# and compare with all the array of ununified labels to make sure they match (use set comparison). if not match, raise an error.
# this is to verify the completeness of the original map.
# 3. set up an optimized map that is map[str,list[(str,str)]], the key is the unified label, the value is a tuple,
# first element is the dataset name (we call it just 'dataset'), second element is the category.


def _ensure_maps(dataset_path: Path) -> None:
    global _OPTIMIZED_MAP
    if _OPTIMIZED_MAP:
        return
    _OPTIMIZED_MAP = {}

    if not _ORIGINAL_MAP:
        return

    dataset_names = download_data.get_all_dataset_name()
    category_to_datasets: Dict[str, List[str]] = {}
    for name in dataset_names:
        try:
            categories = download_data.get_audio_categories(dataset_path, name)
        except FileNotFoundError:
            continue
        for category in categories:
            category_to_datasets.setdefault(category, []).append(name)

    all_known_categories = set(download_data.get_audio_categories(dataset_path, None))
    mapped_categories = {c for cats in _ORIGINAL_MAP.values() for c in cats}
    if all_known_categories != mapped_categories:
        missing = sorted(all_known_categories - mapped_categories)
        extra = sorted(mapped_categories - all_known_categories)
        raise ValueError(f"Original map mismatch: missing={missing}, extra={extra}")

    for label, categories in _ORIGINAL_MAP.items():
        pairs: List[Tuple[str, str]] = []
        for category in categories:
            datasets = category_to_datasets.get(category, [])
            if not datasets:
                raise ValueError(f"Unknown category '{category}' in original map")
            for dataset in datasets:
                pairs.append((dataset, category))
        _OPTIMIZED_MAP[label] = pairs


def get_audio_labels(dataset_path: Path, dataset: Optional[str]) -> list[str]:
    """
    Return the available audio categories for supported datasets.

    Args:
        dataset_path: The directory containing the downloaded datasets.
        dataset: Optional dataset name. When None, all supported datasets are aggregated.

    Returns:
        A sorted list of labels available for the requested dataset(s).
    """
    _ensure_maps(dataset_path)
    if not _ORIGINAL_MAP:
        return []

    if not hasattr(get_audio_labels, "_cache"):
        get_audio_labels._cache = {}
    cache: Dict[Optional[str], List[str]] = get_audio_labels._cache
    cache_key = dataset.strip().lower() if dataset else None
    if cache_key in cache:
        return cache[cache_key]

    if dataset is None:
        result = sorted(_OPTIMIZED_MAP.keys())
        cache[cache_key] = result
        return result

    dataset_name = cache_key
    labels = []
    for label, pairs in _OPTIMIZED_MAP.items():
        if any(pair[0] == dataset_name for pair in pairs):
            labels.append(label)
    result = sorted(labels)
    cache[cache_key] = result
    return result


def get_audio_list_by_label(
    dataset_path: Path, dataset: Optional[str], label: str
) -> list[str]:
    """
    Args:
        dataset_path: the path to the dataset dir
        label: sound class name
        dataset: optional dataset name, if None, search across all datasets
    Returns:
        list of audio file paths (strings)
    """
    _ensure_maps(dataset_path)
    if not _ORIGINAL_MAP:
        return []

    label = label.strip()
    if label not in _ORIGINAL_MAP:
        raise ValueError(f"Unknown label: {label}")

    dataset_name = dataset.strip().lower() if dataset else None
    results: List[str] = []
    for ds_name, ds_category in _OPTIMIZED_MAP[label]:
        if dataset_name is not None and ds_name != dataset_name:
            continue
        results.extend(
            download_data.get_audio_list_by_category(dataset_path, ds_name, ds_category)
        )
    return sorted(results)


def get_all_dataset_name() -> list[str]:
    """
    Returns:
        list of all dataset names
    """
    return download_data.get_all_dataset_name()


def get_rawdata_size():
    total = 0
    for dataset_name in get_all_dataset_name():
        dataset_size = 0
        for label in get_audio_labels(get_raw_dataset_path(), dataset_name):
            dataset_size += len(
                get_audio_list_by_label(
                    get_raw_dataset_path(), dataset=dataset_name, label=label
                )
            )
        total += dataset_size
    return total


def print_all_dataset_info(dataset_path: Path):
    total = 0
    print("Dataset info:")
    for dataset_name in get_all_dataset_name():
        dataset_size = 0
        for label in get_audio_labels(dataset_path, dataset_name):
            dataset_size += len(
                get_audio_list_by_label(dataset_path, dataset=dataset_name, label=label)
            )
        print(f"{dataset_name}: {dataset_size} files")
        total += dataset_size
    print("Total files: ", total)


def print_all_label_info(dataset_path: Path):
    total: Mapping[str, int] = {}
    print("Label info:")
    for dataset_name in get_all_dataset_name():
        for label in get_audio_labels(dataset_path, dataset_name):
            files = get_audio_list_by_label(
                dataset_path, dataset=dataset_name, label=label
            )
            count = len(files)
            total[label] = total.get(label, 0) + count
    total_count = 0
    for label in total:
        total_count += total[label]
        print(f"{label}: {total[label]} files")
    print("Total files:", total_count)
