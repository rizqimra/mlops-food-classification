from roboflow import Roboflow

rf = Roboflow(api_key="BOecATEH3aR0AAhf7q1J")
project = rf.workspace("bangkit").project("indonesian-food-pedsx")
version = project.version(1)
dataset = version.download("folder")
