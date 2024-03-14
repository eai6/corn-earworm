from roboflow import Roboflow
rf = Roboflow(api_key="3mzCICwQ1osUSr2TBrk1")
project = rf.workspace("crosectvision").project("corn-earworm-class-project")
version = project.version(2)
dataset = version.download("yolov8")
