# -*- coding: utf-8 -*-
# @file edit_scene.py
# @brief Edit Scene Directly
# @author sailing-innocent
# @date 2025-05-17
# @version 1.0
# ---------------------------------
import json
from asset_dir import AssetDir

def flatten_json(y):
    out = {}

    def flatten(x, name=''):

        # If the Nested key-value
        # pair is of dict type
        if type(x) is dict:

            for a in x:
                flatten(x[a], name + a + '_')

        # If the Nested key-value
        # pair is of list type
        elif type(x) is list:

            i = 0

            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def unflatten_json(y):
    out = {}

    for a in y:
        keys = a.split('_')
        d = out

        for i in range(len(keys) - 1):
            if keys[i] not in d:
                d[keys[i]] = {}
            d = d[keys[i]]

        d[keys[-1]] = y[a]

    return out


class SceneEdit:
    def __init__(self, scene_path: str):
        self.name = "scene"
        self.scene_path = scene_path
        self.scene_dict = json.load(open(scene_path, "r"))
        self.scene_dict_flat = flatten_json(self.scene_dict)
        # save the scene_dict_flat to a lmdb in memory
        self.geometries = self.scene_dict["__data__"]["geometry_atlas"]["__data__"]["geometries"]
        # print("geometries", self.geometries)
        print("Number of geometries", len(self.geometries)) # 20
        self.attributes = self.scene_dict["__data__"]["geometry_atlas"]["__data__"]["attributes"]
        print("Number of attributes", len(self.attributes))
        self.geometry_slots = self.scene_dict["__data__"]["geometry_slots"]
        print("Number of geometry slots", len(self.geometry_slots))
        self.objects = self.scene_dict["__data__"]["object_collection"]["objects"]
        print("Number of objects", len(self.objects))
        for i, obj in enumerate(self.objects):
            print(f"Object {i}: {obj}")
            obj_geom = obj["geometries"]
            for obj_g in obj_geom:
                print(f"Object {i} Geometry: {obj_g}")
                g_slot = self.geometry_slots[obj_g]["index"]
                print(f"Object {i} Geometry Slot: {g_slot}")
                gg = self.geometries[g_slot]
                print(f"Object {i} Geometry: {gg}")
                g_type = gg["__meta__"]["type"]
                print(f"Object {i} Geometry Type: {g_type}")

                if g_type == "SimplicialComplex":
                    v = gg["__data__"]["vertices"]
                    v_pos_attr_id = v["__data__"]["position"]["index"]
                    v_pos_attr = self.attributes[v_pos_attr_id]
                    tri = gg["__data__"]["triangles"]

                    print(v_pos_attr)
                else:
                    print(f"Object {i} Geometry Type: {g_type} is not SimplicialComplex")

    def __call__(self, pattern: str):
        """
        :param pattern: the pattern to search for
        :return: a list of objects that match the pattern
        """
        result = []
        for k, v in self.scene_dict_flat.items():
            if pattern in k:
                result.append((k, v))
        return result


    def to_json(self):
        return unflatten_json(self.scene_dict_flat)

if __name__ == "__main__":
    folder = AssetDir.folder(__file__)
    scene_path = f"{folder}/scene.json"
    scene_edit = SceneEdit(scene_path)
    # objects = scene_edit("objectsd")
    # geometries = scene_edit("geometry_atlas___data___geometries")
    # print(geometries)

