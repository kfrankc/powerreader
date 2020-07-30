# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import webvtt as wvt

caption_path = 'input/captions.vtt'


def get_sec(time_str):
    """Get Seconds from time."""
    t, f = time_str.split('.')
    h, m, s = t.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(f) * 0.001


def get_slide_timestamps():
    slide_timestamps = []
    current_slide = 1
    start_time = 0
    for caption in wvt.read(caption_path):
        if("next slide" in caption.text.lower()):
            slide_timestamps.append((current_slide, start_time, get_sec(caption.end)))
            current_slide += 1
            start_time = get_sec(caption.end)
        if("previous slide" in caption.text.lower()):  # Have empty section if next and previous both happen
            slide_timestamps.append((current_slide, start_time, get_sec(caption.end)))
            current_slide = max(current_slide - 1, 1)
            start_time = get_sec(caption.end)
    return slide_timestamps


# Use a test list input of (slide number, start time, end time) list of tuples
slide_timestamp = [("1", "2", "22"), ("2", "22", "48"), ("3", "50", "80"), ("4", "80", "98")]
# for i in slide_timestamp:
#     print(i)


def generate_mapping(slide_timestamp, captions):
    '''
    Input: list of (slide#, start, end), captions vtt file
    Output: list of (slide#, caption)
    '''
    mapping = {}
    for caption in wvt.read(captions):
        for st_map in slide_timestamp:
            if get_sec(caption.start) >= int(st_map[1]) and get_sec(caption.end) <= int(st_map[2]):
                # caption belongs to this slide
                if st_map[0] in mapping.keys():
                    temp = mapping[st_map[0]]
                    mapping[st_map[0]] = temp + ' ' + caption.text
                    break
                else:
                    mapping[st_map[0]] = caption.text
                    break
    print(mapping)
    return mapping


# Main
slide_timestamps = get_slide_timestamps()
output = generate_mapping(slide_timestamps, caption_path)

# Write to text file

f = open('output/mappedcaptions.txt', 'wt')
for key, value in output.items():
    f.write('{}: {}\n'.format(key, value.replace('\n', ' ')))
f.close()
# data = str(output)
# f.write(data)
# f.close()

# # Exploratory
# for caption in wvt.read('scripts/test_captions.vtt'):
#     print(caption.start)
#     print(caption.end)
#     print(caption.text)

# Output:
# list of slide number -> text dictionary
