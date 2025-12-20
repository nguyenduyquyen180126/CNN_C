import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def load_txt(file_path):
    with open(file_path, 'r') as f:
        # Đọc dòng đầu tiên để lấy kích thước tensor
        # Nếu dòng đầu chi có 1 số, cho số đố là H, còn lại là 1
        first_line = f.readline().strip()
        dims = [int(x) for x in first_line.split(',') if x.strip()]
        if len(dims) == 1:
            N, C, H, W = dims[0], 1, 1, 1
        if len(dims) == 2:
            N, C, H, W = dims[0], dims[1], 1, 1
        elif len(dims) == 4:
            N, C, H, W = dims
        else:
            raise ValueError(f"Dong dau khong hop le {len(dims)}: {dims}")
        # Tách chuỗi thành các số (bỏ qua các khoảng trắng/ô trống)
        raw_text = f.read().replace('\n', ',')
        flat_data = [float(x) for x in raw_text.split(',') if x.strip()]

    
    tensor_nhwc = np.zeros((N, C, H, W), dtype=np.float32)

    # Đúng theo thứ tự ghi trong code C
    idx = 0
    for n in range(N):
        for c in range(C):    
            for h in range(H):
                for w in range(W):
                    
                    # Lấy giá trị từ danh sách phẳng
                    val = flat_data[idx]
                    tensor_nhwc[n, c, h, w] = val
                    
                    idx += 1

    return tensor_nhwc
def sosanh(tensor1, tensor2):

    diff = np.abs(tensor1 - tensor2)
    
    flat_diff = diff.flatten()
    
    plt.figure(figsize=(12, 6)) # Tạo khung hình rộng để dễ nhìn
    
    plt.plot(flat_diff, label='Absolute Error', color='red', linewidth=0.8)


 
    plt.title(f"Biểu đồ sai số theo vị trí tuyệt đối (Max Error: {np.max(diff):.6f})")
    plt.xlabel("Vị trí tuyệt đối (Absolute Index)")
    plt.ylabel("Giá trị sai số (Abs Diff)")
    plt.grid(True, linestyle='--', alpha=0.7) # Thêm lưới để dễ dóng giá trị
    plt.legend()
    

    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model output')
    parser.add_argument('--golden', type=str, required=True, help='Path to golden output txt file')
    parser.add_argument('--model_out', type=str, required=True, help='Path to model output txt file')
    args = parser.parse_args()
    golden = load_txt(args.golden)
    model_output = load_txt(args.model_out)
    sosanh(golden, model_output)