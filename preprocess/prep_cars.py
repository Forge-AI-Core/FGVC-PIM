import os
import shutil
import scipy.io

def prep_stanford_cars():
    base_dir = "datas/Stanford-Cars"
    
    print("🛠️ Stanford Cars 데이터셋 전처리 및 기종별 폴더 구조화 시작...")
    
    # 이미 올바른 매트릭스 파일(185KB)로 덮어쓰기 되어 있으므로 바로 파싱 진행
    splits = [
        {"name": "train", "img_dir": "cars_train", "mat_file": "cars_train_annos.mat"},
        {"name": "test", "img_dir": "cars_test", "mat_file": "cars_test_annos.mat"}
    ]
    
    for split in splits:
        mat_path = os.path.join(base_dir, "devkit", split["mat_file"])
        img_src_dir = os.path.join(base_dir, split["img_dir"])
        
        # 압축 해제 방식에 따라 이중 폴더(cars_train/cars_train) 구조가 된 경우 자동 감지
        nested_dir = os.path.join(img_src_dir, split["img_dir"])
        if os.path.exists(nested_dir):
            img_src_dir = nested_dir
            
        out_dir = os.path.join(base_dir, split["name"])
        
        if not os.path.exists(mat_path) or not os.path.exists(img_src_dir):
            print(f"❌ 소스 경로를 찾을 수 없어 {split['name']} 분류를 건너뜁니다:\n   👉 {mat_path} 또는 {img_src_dir}")
            continue
            
        print(f"\n🚗 {split['name']} 데이터셋 이미지 기종별 폴더 분류 중...")
        mat_data = scipy.io.loadmat(mat_path)
        annotations = mat_data['annotations'][0]
        
        os.makedirs(out_dir, exist_ok=True)
        
        count = 0
        for anno in annotations:
            # .mat 구조체 배열에서 파일명과 클래스 번호(1~196) 안전 추출
            fname = str(anno['fname'].flat[0])
            class_id = int(anno['class'].flat[0])
            
            # 3자리 숫자 포맷으로 폴더명 생성 (예: 001, 002, ..., 196)
            class_dir = os.path.join(out_dir, f"{class_id:03d}")
            os.makedirs(class_dir, exist_ok=True)
            
            src_img = os.path.join(img_src_dir, fname)
            dst_img = os.path.join(class_dir, fname)
            
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img)
                count += 1
                
        print(f"✅ {split['name']}셋 기종별 폴더 구조화 완료! (총 {count}장 신규 복사됨)\n   👉 저장 경로: {out_dir}")

if __name__ == "__main__":
    prep_stanford_cars()
