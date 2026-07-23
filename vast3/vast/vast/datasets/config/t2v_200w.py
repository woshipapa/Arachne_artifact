import itertools

person = dict(
    sports=[
        "/root/datasets/Text2Video/annotations/200w/pack_20_1_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_20_1_2.json",
        "/root/datasets/Text2Video/annotations/200w/pack_20_2_new.json",
    ],
    documentary=[
        "/root/datasets/Text2Video/annotations/200w/pack_36_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_36_2.json",
        "/root/datasets/Text2Video/annotations/200w/pack_36_3.json",
        "/root/datasets/Text2Video/annotations/200w/pack_36_4.json",
        "/root/datasets/Text2Video/annotations/200w/pack_36_5.json",
        "/root/datasets/Text2Video/annotations/200w/pack_36_6.json",
        "/root/datasets/Text2Video/annotations/200w/pack_36_7.json",
    ],
    ad=[
        "/root/datasets/Text2Video/annotations/200w/pack_66_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_66_2.json",
    ],
    # 艺术类
    art=[
        "/root/datasets/Text2Video/annotations/200w/pack_67_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_67_2.json",
        "/root/datasets/Text2Video/annotations/200w/pack_67_3.json",
        "/root/datasets/Text2Video/annotations/200w/pack_67_4.json",
    ],
    # 其他类
    other=[
        "/root/datasets/Text2Video/annotations/200w/pack_55_1_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_1_2.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_2_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_2_2.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_3_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_3_2.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_4_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_4_2.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_5_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_5_2.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_6_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_6_2.json",
        "/root/datasets/Text2Video/annotations/200w/pack_55_7.json",
        "/root/datasets/Text2Video/annotations/200w/pack_71_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_71_2.json",
        # 未知
        "/root/datasets/Text2Video/annotations/200w/pack_zwzx_1.json",
        "/root/datasets/Text2Video/annotations/200w/pack_zwzx_2.json",
        "/root/datasets/Text2Video/annotations/200w/pack_zwzx_3.json",
    ],
)
no_person = dict(
    sport=[
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_pack_20_2.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_pack_20_1_2.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_pack_20_1_1.json",
    ],
    other=[
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_zwzx_1_slice_new_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_7_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_1_2_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_4_2_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_1_1_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_2_1_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_5_1_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_6_2_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_6_1_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_3_1_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_4_1_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_2_2_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_3_2_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_55_5_2_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_71_1_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_71_2_slice_0.json",
    ],
    ad=[
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_66_2_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_66_1_slice_0.json",
    ],
    documentary=[
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_36_7_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_36_1_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_36_5_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_36_6_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_36_3_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_36_2_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_36_4_slice_0.json",
    ],
    art=[
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_67_4_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_67_3_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_67_1_slice_0.json",
        "/root/datasets/Text2Video/annotations/200w_nobody/pack_67_2_slice_0.json",
    ],
)

all_data_list = list(person.values()) + list(no_person.values())

all_data_list = list(itertools.chain(*all_data_list))
from typing import Optional, List


def get_data_list(video_type_list: Optional[List] = None, only_persion=False):
    if video_type_list is None:
        return all_data_list
    else:
        data_list = []
        for video_type in video_type_list:
            data_list += person.get(video_type, [])
            if not only_persion:
                data_list += no_person.get(video_type, [])
    return data_list
