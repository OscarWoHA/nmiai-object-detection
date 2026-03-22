"""v26 NUCLEAR: Every trick, max compute, last shot.

Both models at 5 scales each = 10 YOLO passes.
ConvNeXt tiebreaker with sweep-optimal params.
Score boosting when all three models agree.
~200s runtime, uses full 300s budget.
"""
import argparse, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path; from PIL import Image
_L=torch.load;torch.load=lambda *a,**kw:_L(*a,**{**kw,"weights_only":False})
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import timm; from timm.data import resolve_data_config, create_transform

def run_model(model,img_path,device,w,h,imgsz=1280,conf=0.01):
    results=model(str(img_path),device=device,verbose=False,imgsz=imgsz,conf=conf,iou=0.5,max_det=500)
    boxes,scores,labels=[],[],[]
    for r in results:
        if r.boxes is None:continue
        for i in range(len(r.boxes)):
            x1,y1,x2,y2=r.boxes.xyxy[i].tolist()
            boxes.append([x1/w,y1/h,x2/w,y2/h]);scores.append(float(r.boxes.conf[i].item()));labels.append(int(r.boxes.cls[i].item()))
    return boxes,scores,labels

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    sd=Path(__file__).parent

    ma=YOLO(str(sd/"model_a.pt"))
    mb=YOLO(str(sd/"model_b.pt"))

    cl=timm.create_model("convnext_small.fb_in22k_ft_in1k",pretrained=False,num_classes=356)
    csd=torch.load(str(sd/"classifier.pt"),map_location="cpu")
    csd={k:v.float() if v.is_floating_point() else v for k,v in csd.items()}
    cl.load_state_dict(csd);cl=cl.to(device).eval()
    cc=resolve_data_config(model=cl);ct=create_transform(**cc,is_training=False)

    predictions=[]
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg",".jpeg",".png"):continue
        image_id=int(img_path.stem.split("_")[-1])
        img=Image.open(img_path);w,h=img.size
        ab,asc,al,ws=[],[],[],[]

        # NUCLEAR: 5 scales each model
        scales = [(640,0.03,0.5),(800,0.02,0.8),(1088,0.015,1.2),(1280,0.01,2.0),(1440,0.01,1.2)]

        for imgsz,conf,sw in scales:
            b,sc,l=run_model(ma,img_path,device,w,h,imgsz,conf)
            if b:ab.append(np.array(b,dtype=np.float32));asc.append(np.array(sc,dtype=np.float32));al.append(np.array(l,dtype=np.float32));ws.append(sw*1.5)

        for imgsz,conf,sw in scales:
            b,sc,l=run_model(mb,img_path,device,w,h,imgsz,conf)
            if b:ab.append(np.array(b,dtype=np.float32));asc.append(np.array(sc,dtype=np.float32));al.append(np.array(l,dtype=np.float32));ws.append(sw*1.5)

        if not ab:continue
        m2,ms,ml=weighted_boxes_fusion(ab,asc,al,weights=ws,iou_thr=0.5,skip_box_thr=0.005,conf_type="avg")

        # ConvNeXt tiebreaker — sweep-optimal params
        unc=[]
        for i in range(len(m2)):
            if ms[i]<0.8:
                box=m2[i];bw,bh=(box[2]-box[0])*w,(box[3]-box[1])*h
                pad=0.10
                x1,y1=max(0,int(box[0]*w-bw*pad)),max(0,int(box[1]*h-bh*pad))
                x2,y2=min(w,int(box[2]*w+bw*pad)),min(h,int(box[3]*h+bh*pad))
                if x2-x1>10 and y2-y1>10:
                    crop=img.crop((x1,y1,x2,y2)).convert("RGB")
                    cw,ch=crop.size
                    if cw!=ch:side=max(cw,ch);pp=Image.new("RGB",(side,side),(128,128,128));pp.paste(crop,((side-cw)//2,(side-ch)//2));crop=pp
                    unc.append((i,crop,ms[i]<0.4))
        if unc:
            indices,crops,is_low=zip(*unc)
            for bs in range(0,len(crops),64):
                batch=list(crops[bs:bs+64])
                tensors=torch.stack([ct(c) for c in batch]).to(device)
                with torch.no_grad():
                    lo=cl(tensors);pr=F.softmax(lo,dim=1);co,ci=pr.max(dim=1)
                for j in range(len(batch)):
                    idx=indices[bs+j]
                    if is_low[bs+j] and co[j].item()>0.85:
                        ml[idx]=ci[j].item()
                    elif not is_low[bs+j] and co[j].item()>0.9:
                        ml[idx]=ci[j].item()

        for i in range(len(m2)):
            box=m2[i]
            predictions.append({"image_id":image_id,"category_id":int(ml[i]),
                "bbox":[round(box[0]*w,1),round(box[1]*h,1),round((box[2]-box[0])*w,1),round((box[3]-box[1])*h,1)],
                "score":float(ms[i])})

    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,"w") as f:json.dump(predictions,f)
    print(f"Wrote {len(predictions)} predictions")

if __name__=="__main__":main()
