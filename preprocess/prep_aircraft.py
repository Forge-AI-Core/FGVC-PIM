import os
import pandas as pd
import shutil

base_dir = "datas/FGVC-Aircraft"
img_dir = os.path.join(base_dir, "fgvc-aircraft-2013b", "data", "images")

print("분류 시작...")
for split in ["train", "val", "test"]:
    csv_path = os.path.join(base_dir, f"{split}.csv")
    if not os.path.exists(csv_path): continue
    df = pd.read_csv(csv_path)
    
    out_dir = os.path.join(base_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        filename = str(row['filename'])
        if not filename.endswith('.jpg'): filename += '.jpg'
        
        # 폴더명에 공백이나 / 가 있으면 오류날 수 있으니 변환
        class_name = str(row['Classes']).replace('/', '_')
        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        src = os.path.join(img_dir, filename)
        dst = os.path.join(class_dir, filename)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

print("✅ Aircraft 데이터 폴더 분류 완료!")
