# Scene Commit

This example has 2 parts:

- `server_run.py`
- `client_get.py`

`server_run.py` creates a scene and saves the `SceneSnapshot` to the file. Every time it `advance()`, it will save a `SceneSnapshotCommit` to a file, which can be used to update a `Scene`.

`client_get.py` loads the scene from a `SceneSnapshot` file, and updates the scene with the `SceneSnapshotCommit` file per frame.

This facility can cover the demands of `Client-Server` communication.