from roboflow import Roboflow
rf = Roboflow(api_key="r2shRVOTAqV23q8DRzSj")
project = rf.workspace("insectnet-2024").project("corn-earworn-class-project")
version = project.version(1)
dataset = version.download("yolov8")
