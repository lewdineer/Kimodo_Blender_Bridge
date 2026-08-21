import json, os

def post_process_motion(local_rot_mats, root_positions, foot_contacts, skeleton, constraint_lst=None):
    path = os.environ.get("ARDY_MOCK_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps({"kind": "postprocess",
                                 "frames": int(local_rot_mats.shape[1]),
                                 "constraints": len(constraint_lst or [])}) + "\n")
    return {"local_rot_mats": local_rot_mats, "root_positions": root_positions}
