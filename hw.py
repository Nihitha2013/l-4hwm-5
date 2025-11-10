import os
import requests
from config import HF_API_KEY

def main():
  folder_Path=input("Enter the folder path(press Enter for 'images'): ").strip()
  if not folder_Path:
    folder_Path="images"

  if not os.path.isdir(folder_Path):
     print(f"Folder '{folder_Path}' does not exist.")
     return

  MODEL_ID="nlpconnect/vit-gpt2-image-captioning"
  API_URL=f"https://api-inference.huggingface.co/models/{MODEL_ID}"

  headers={"Authorization":f"Bearer {HF_API_KEY}"}

  image_files=[f for f in os.listdir(folder_Path) 
               if f.lower().endswith(('.png','.jpg','.jpeg','.gif','.bmp'))]
  if not image_files:
     print(f"No image files found in folder '{folder_Path}'.")
     return

  captions=[]
  for img_name in image_files:
     img_path=os.path.join(folder_Path,img_name)
     print(f"\nProcessing image '{img_path}'...")

     try:
       with open(img_path,"rb")as img_file:
         image_bytes=img_file.read()
     except Exception as e:
         print(f"Error reading image file '{img_name}'Error:{e}")
         continue

     try:
         response=requests.post(API_URL,headers=headers,data=image_bytes)
         result=response.json()
     except requests.exceptions.RequestException as req_e:
        print(f"Network error while proccessing '{img_name}':{req_e}")
        continue

     if isinstance(result,dict) and"error" in result:
        print(f"[API Error] {result['error']} for '{img_name}'")
        continue

     caption=result[0].get("generated_text","No caption found")
     print(f"Caption: {caption}")
     captions.append((img_name,caption))

  if captions:
     summary_file=os.path.join(folder_Path,"captions_summary.txt")
     with open(summary_file,"w",encoding='utf-8') as sf:
      for img_name,caption in captions:
         sf.write(f"{img_name}:{caption}\n")
     print(f"\nCaptions summary saved to '{summary_file}'")
  else:
     print("\nNo captions were generated.")

if __name__=="__main__":
   main()
       
