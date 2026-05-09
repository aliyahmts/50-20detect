from ultralytics import YOLO

model = YOLO('my_model.pt')
print("Classes in model:")
for idx, name in model.names.items():
    print(f"{idx}: {name}")