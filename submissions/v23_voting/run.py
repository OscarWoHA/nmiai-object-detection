"""v23: YOLO ensemble + ConvNeXt VOTING on ALL boxes.

For every detection, run ConvNeXt. Final class = weighted vote:
- YOLO class gets weight = yolo_conf
- ConvNeXt class gets weight = conv_conf * 0.5
If they agree, confidence goes up. If they disagree, lower conf wins less.
"""
import argparse, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
_L = torch.load
torch.load = lambda *a, **kw: _L(*a, **{**kw, "weights_only": False})
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import timm
from timm.data import resolve_data_config, create_transform

def run_model(model, img_path, device, w, h, imgsz=1280, conf=0.01):
    results = model(str(img_path), device=device, verbose=False, imgsz=imgsz, conf=conf, iou=0.5, max_det=500)
    boxes, scores, labels = [], [], []
    for r in results:
        if r.boxes is None: continue
        for i in range(len(r.boxes)):
            x1,y1,x2,y2 = r.boxes.xyxy[i].tolist()
            boxes.append([x1/w,y1/h,x2/w,y2/h])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))
    return boxes, scores, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = Path(__file__).parent

    model_a = YOLO(str(sd / "model_a.pt"))
    model_b = YOLO(str(sd / "model_b.pt"))

    classifier = timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=False, num_classes=356)
    classifier.load_state_dict(torch.load(str(sd / "classifier.pt"), map_location="cpu"))
    classifier = classifier.to(device).eval()
    cls_config = resolve_data_config(model=classifier)
    cls_transform = create_transform(**cls_config, is_training=False)

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg",".jpeg",".png"): continue
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path); w,h = img.size
        ab,asc,al,ws = [],[],[],[]

        for model,mw in [(model_a,1.5),(model_b,1.5)]:
            for imgsz,conf,sw in [(960,0.02,1.0),(1280,0.01,2.0),(1440,0.01,1.5)]:
                boxes,scores,labels = run_model(model,img_path,device,w,h,imgsz,conf)
                if boxes:
                    ab.append(np.array(boxes,dtype=np.float32))
                    asc.append(np.array(scores,dtype=np.float32))
                    al.append(np.array(labels,dtype=np.float32))
                    ws.append(sw*mw)

        if not ab: continue
        mb,ms,ml = weighted_boxes_fusion(ab,asc,al,weights=ws,iou_thr=0.5,skip_box_thr=0.005,conf_type="avg")

        # Crop ALL detections for ConvNeXt voting
        crops = []
        valid = []
        for i in range(len(mb)):
            box = mb[i]
            bw,bh = (box[2]-box[0])*w,(box[3]-box[1])*h
            pad = 0.12
            x1,y1 = max(0,int(box[0]*w-bw*pad)),max(0,int(box[1]*h-bh*pad))
            x2,y2 = min(w,int(box[2]*w+bw*pad)),min(h,int(box[3]*h+bh*pad))
            if x2-x1>10 and y2-y1>10:
                crop = img.crop((x1,y1,x2,y2)).convert("RGB")
                cw,ch = crop.size
                if cw!=ch:
                    side=max(cw,ch)
                    p=Image.new("RGB",(side,side),(128,128,128))
                    p.paste(crop,((side-cw)//2,(side-ch)//2))
                    crop=p
                crops.append(crop)
                valid.append(i)

        # Batch classify ALL crops
        if crops:
            for bs in range(0,len(crops),64):
                batch = crops[bs:bs+64]
                tensors = torch.stack([cls_transform(c) for c in batch]).to(device)
                with torch.no_grad():
                    logits = classifier(tensors)
                    probs = F.softmax(logits,dim=1)
                    confs,cls_ids = probs.max(dim=1)
                for j in range(len(batch)):
                    idx = valid[bs+j]
                    yolo_cls = int(ml[idx])
                    yolo_conf = float(ms[idx])
                    conv_cls = cls_ids[j].item()
                    conv_conf = confs[j].item()

                    # Weighted voting
                    if yolo_cls == conv_cls:
                        # Agreement — keep YOLO class, boost confidence slightly
                        pass
                    else:
                        # Disagreement — let higher confidence win
                        yolo_weight = yolo_conf
                        conv_weight = conv_conf * 0.4  # ConvNeXt gets less trust
                        if conv_weight > yolo_weight:
                            ml[idx] = conv_cls

        for i in range(len(mb)):
            box = mb[i]
            predictions.append({"image_id":image_id,"category_id":int(ml[i]),
                "bbox":[round(box[0]*w,1),round(box[1]*h,1),round((box[2]-box[0])*w,1),round((box[3]-box[1])*h,1)],
                "score":float(ms[i])})

    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,"w") as f: json.dump(predictions,f)
    print(f"Wrote {len(predictions)} predictions")

if __name__=="__main__": main()
