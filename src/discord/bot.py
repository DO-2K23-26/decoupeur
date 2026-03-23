import os, io, re, cv2, torch, numpy as np, discord
from PIL import Image
import segmentation_models_pytorch as smp
import torch.nn as nn
                                                                                                                                       
# ── Config ──                                                                                                                           
CKPT_PATH = "./paupau/checkpoints/best.pth"
IMG_SIZE = 512                                                                                                                           
MEAN = np.array([0.485, 0.456, 0.406])                                                                                                   
STD = np.array([0.229, 0.224, 0.225])                                                                                                    
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]  # à définir avant de lancer
SENTIMENT_CKPT_PATH = "./checkpoints/best_sentiment.pth"
SENTIMENT_CHANNEL_ID = REPLACEHERE
SEGMENTATION_CHANNEL_ID = REPLACEHERE
MAX_LEN = 256

# ── Tokenizer pour le sentiment ──
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zàâçéèêëïîôùûüÿñæœ0-9\s]', ' ', text)
    return text.split()

# ── Modèle Sentiment (BiLSTM) ──
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden))

# ── Chargement des modèles ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Segmentation
seg_model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=1, activation=None).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
seg_model.load_state_dict(ckpt["state_dict"])
seg_model.eval()
print(f"Modèle segmentation chargé sur {device}")

# Sentiment
sent_ckpt = torch.load(SENTIMENT_CKPT_PATH, map_location=device)
vocab = sent_ckpt["vocab"]
sent_model = BiLSTMClassifier(
    vocab_size=len(vocab),
    embed_dim=sent_ckpt["embed_dim"],
    hidden_dim=sent_ckpt["hidden_dim"],
    num_classes=2,
).to(device)
sent_model.load_state_dict(sent_ckpt["model_state_dict"])
sent_model.eval()
print(f"Modèle sentiment chargé sur {device}")
                                                                                                                                       
# ── Fonction d'inférence (copie de ton notebook) ──                                                                                     
def infer_bytes(img_bytes: bytes) -> bytes:                                                                                              
  """Prend des bytes d'image, renvoie un PNG transparent détouré."""                                                                   
  arr = np.frombuffer(img_bytes, np.uint8)                                                                                             
  bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)                                                                                            
  h, w = bgr.shape[:2]                                                                                                                 
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)                                                                                           
                                                                                                                                       
  t = torch.tensor(
      ((cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)) / 255.0 - MEAN) / STD)                                                                   
      .transpose(2, 0, 1),                                                                                                             
      dtype=torch.float32,
  ).unsqueeze(0).to(device)                                                                                                            
              
  with torch.no_grad():
      pred = torch.sigmoid(seg_model(t))[0, 0].cpu().numpy()
                                                                                                                                       
  mask = cv2.resize(pred, (w, h))                                                                                                      
  mask = (mask > 0.5).astype(np.uint8) * 255                                                                                           
                                                                                                                                       
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                                                     
  if contours:
      mask = np.zeros_like(mask)                                                                                                       
      cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
                                                                                                                                       
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)                                                                               
  mask = cv2.GaussianBlur(mask, (5, 5), 0)                                                                                             

  # PNG transparent                                                                                                                    
  result = Image.fromarray(np.dstack([rgb, mask]), "RGBA")
  buf = io.BytesIO()                                                                                                                   
  result.save(buf, format="PNG")
  buf.seek(0)
  return buf.getvalue()                                                                                                                

# ── Inférence sentiment ──
def infer_sentiment(text):
    """Prédit le sentiment d'un texte. Renvoie (label, prob_neg, prob_pos)."""
    tokens = tokenize(text)
    indices = [vocab.get(t, vocab.get('<unk>', 1)) for t in tokens[:MAX_LEN]]
    if len(indices) == 0:
        indices = [vocab.get('<unk>', 1)]
    length = len(indices)
    if length < MAX_LEN:
        indices += [0] * (MAX_LEN - length)

    x = torch.tensor([indices], dtype=torch.long).to(device)
    l = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        output = sent_model(x, l)
        probs = torch.softmax(output, dim=1).cpu().squeeze().numpy()

    labels = ['Négatif', 'Positif']
    return labels[probs.argmax()], probs[0], probs[1]

# ── Bot Discord ──
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
                                                                                                                                       
@client.event
async def on_ready():                                                                                                                    
  print(f"Bot connecté : {client.user}")

@client.event
async def on_message(message):
  if message.author.bot:
      return

  # ── Sentiment Analysis ──
  if message.channel.id == SENTIMENT_CHANNEL_ID and message.content.strip():
      label, prob_neg, prob_pos = infer_sentiment(message.content)
      emoji = "👍" if label == "Positif" else "👎"
      await message.reply(
          f"{emoji} **{label}**\n"
          f"Négatif : {prob_neg:.1%} / Positif : {prob_pos:.1%}"
      )
      return

  # ── Segmentation ──
  if message.channel.id != SEGMENTATION_CHANNEL_ID:
      return

  for attachment in message.attachments:                                                                                               
      if not attachment.content_type or not attachment.content_type.startswith("image/"):                                              
          continue                                                                                                                     

      await message.add_reaction("⏳")                                                                                                 
      img_bytes = await attachment.read()
      result_bytes = infer_bytes(img_bytes)                                                                                            
      await message.remove_reaction("⏳", client.user)                                                                                 
                                                                                                                                           
      try:                                                                                                                             
          file = discord.File(io.BytesIO(result_bytes), filename="detourage.png")                                                      
          await message.reply(file=file)                                                                                               
      except discord.HTTPException:                                                                                                    
          await message.reply("L'image détourée est trop lourde pour être envoyée (limite Discord : 25 Mo).")       

client.run(DISCORD_TOKEN)
