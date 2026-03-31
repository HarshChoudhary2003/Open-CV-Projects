import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ReIDModel:
    def __init__(self, threshold=0.8):
        # We use a ResNet50 pretrained as our feature extractor
        # In a real-world scenario, you'd load a model specifically trained on Market1501 or similar (like FastReID or TorchReID)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the classification head to get embeddings
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.threshold = threshold
        self.embeddings_db = {} # Maps global person_id to embedding vector

    def extract_features(self, person_img_cv2):
        if person_img_cv2 is None or person_img_cv2.size == 0:
            return None
            
        img_pil = Image.fromarray(person_img_cv2[..., ::-1]) # BGR to RGB
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_tensor).cpu().numpy().flatten()
            
        # Normalize to unit vector for cosine similarity
        features = features / np.linalg.norm(features)
        return features

    def match_identity(self, embedding, local_track_id, camera_id):
        """
        Returns the matched global person_id or creates a new one.
        """
        if embedding is None:
            return f"Unknown_{local_track_id}"

        best_match_id = None
        best_similarity = -1

        for person_id, stored_embedding in self.embeddings_db.items():
            similarity = np.dot(embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id

        if best_similarity > self.threshold:
            return best_match_id
        else:
            # Register new person
            new_id = f"Person_{len(self.embeddings_db) + 1}"
            self.embeddings_db[new_id] = embedding
            return new_id
