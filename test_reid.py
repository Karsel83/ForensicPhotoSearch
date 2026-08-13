from reid_model import PersonReID
#Embedding 테스트

model = PersonReID()

image_path = "data/person_crops/human1_person_0.jpg"

embedding = model.extract(image_path)

print()
print("[*] Embedding 생성 완료")
print("[*] Shape:", embedding.shape)
print("[*] Vector:")
print(embedding)