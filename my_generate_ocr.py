

font_dir_path = '/Users/apple/text_renderer2/text_renderer/font/'
bg_dir_path = '/Users/apple/text_renderer2/text_renderer/bg/'

import random
from PIL import Image, ImageFont, ImageDraw
import glob
from fontTools.ttLib import TTFont

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor
import numpy as np
import json
from tqdm.auto import tqdm
import csv
from torchvision.transforms import v2 as T
from PIL import ImageFile
import wget
from transformers import AutoTokenizer


ImageFile.LOAD_TRUNCATED_IMAGES = True


# TrOCR processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
processor.image_processor.size['height'] = 32
processor.image_processor.size['width'] = 128
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small", use_fast=False)

black_list_fonts = [
    'NINJAL Hentaigana',
    'yokumiruHentaikanaGothicLight',
    'yokumiruHentaikanaGothic',
    'yokumiruHentaikanaGothicRegular',
    'yokumiruHentaikanaGothicMedium',
]


def get_all_chars(b, e):
    return [chr(i) for i in range(ord(b), ord(e) + 1)]


def has_glyph(font, glyph):
    for table in font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False

class DataGenerator(Dataset):
  def __init__(self, ):
    super().__init__()
    self.hiragana = get_all_chars('ぁ', 'ゟ')
    self.katakana = get_all_chars('゠', 'ヿ')
    self.kanji, self.kanji_w = self.init_kanji()
    self.symbols_punct = get_all_chars('、', u'\u303F') + get_all_chars(u'\u31F0', u'\u31FF') + get_all_chars(u'\uFF5F', '･')
    self.ascii_char = get_all_chars(' ', '~')

    self.char_list = [self.hiragana, self.katakana, self.kanji, self.symbols_punct, self.ascii_char]
    self.sampleList = [0, 1, 2, 3, 4]
    self.weights=(10, 10, 50, 10, 10.5)
    self.max_length = 40
    self.max_font_size = 40
    self.max_deg = 4
    self.init_bg()
    self.init_fonts()
        
    self.crop_bg = T.RandomCrop((64,64))
    self.img_transforms = T.RandomApply(torch.nn.ModuleList([
            T.ColorJitter(),
             T.GaussianBlur(3)
         ]), p=0.3)

  def init_kanji(self):
    char_freq = {}
    if not os.path.exists('wikipedia_characters.csv'):
        wget.download('https://github.com/scriptin/kanji-frequency/raw/master/data/wikipedia_characters.csv')
    with open('wikipedia_characters.csv', 'r' ) as f:
      reader = csv.reader(f)
      for i, row in enumerate(reader):
        if i==0:
          continue
        if i==1:
          total_count = int(row[3])
          continue
        char_freq[row[2]] = int(row[3])/total_count
    kanji = list(char_freq.keys())
    kanji_w = np.array(list(char_freq.values()))
    kanji_w = np.sqrt(kanji_w)
    return kanji, kanji_w

  def init_fonts(self):
    self.all_font_paths = glob.glob(font_dir_path + '*')
    self.fonts_check = []
    self.font_vocabs = []
    # print(self.all_font_paths)
    for font_path in tqdm(self.all_font_paths):
      font = TTFont(font_path)
      font2 = ImageFont.truetype(font_path, self.max_font_size)
      if font2.font.family in black_list_fonts:
        continue
      self.fonts_check.append(font)
      self.font_vocabs.append(self.get_font_vocab(font, font2))
      # break
    

  def get_font_vocab(self, font_check, font):
    font_vocab = set()
    for char_subtype in self.char_list:
      for char in char_subtype:
        if has_glyph(font_check, char) and font.getmask(char).size[1] != 0:
          font_vocab.add(char)
    return font_vocab


  def init_bg(self):
    self.bgs = []
    all_bg_paths = glob.glob(bg_dir_path+'*/*')
    for bg_path in all_bg_paths:
      bg = Image.open(bg_path)
      self.bgs.append(bg)

  def get_random_font(self):
    font_size = random.randint(30, self.max_font_size)
    font_index = random.randint(0, len(self.fonts_check)-1)
    font_path = self.all_font_paths[font_index]
    font_vocab = self.font_vocabs[font_index]
    font = ImageFont.truetype(font_path, font_size)
    return font, font_vocab

  def get_random_bg(self):
    bg = random.choice(self.bgs)
    bg = self.crop_bg(bg)
    return bg

  def get_random_deg(self):
    return random.randint(-self.max_deg, self.max_deg)

  def get_random_padding(self, width, height):
    l_pad = int(random.randint(0, 10)/100*width)
    t_pad = int(random.randint(0, 10)/100*height)
    r_pad = int(random.randint(0, 10)/100*width)
    b_pad = int(random.randint(0, 10)/100*height)
    return (l_pad, t_pad, r_pad, b_pad)


  def get_random_color(self):
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)

  def get_random_text(self, length, font_vocab=None):
    char_indexes = random.choices(self.sampleList, weights=self.weights, k=length)
    text = ''
    for char_index in char_indexes:
      while True:
        if char_index == 2: # kanji
          char = random.choices(self.kanji, weights=self.kanji_w)[0]
        else:
          char = random.choice(self.char_list[char_index])[0]
        if char in font_vocab:
          break
        char_index = random.choices(self.sampleList, weights=self.weights)[0]
      text += random.choice(char)
    return text


  def __len__(self):
    return int(1e3)

  def __getitem__(self, idx):
    while True:
      try:
        length = random.randint(1, self.max_length)
        font, font_vocab = self.get_random_font()
        text = self.get_random_text(length, font_vocab)
        

        text_color = self.get_random_color()
        bg = self.get_random_bg().convert("RGB")

        text2 = " ".join(text) if not random.randint(0, 10) > 0 else text
        text_size = font.getsize(text2)
        padding = self.get_random_padding(text_size[0], text_size[1])
        size = (text_size[0] + padding[0] + padding[2], text_size[1] + padding[1] + padding[3])

        mask_text = Image.new('RGBA', size)
        draw = ImageDraw.Draw(mask_text)
        
        draw.text(( padding[0], padding[1]), text=text2, font = font, align ="left", fill=text_color)
        mask_text=mask_text.rotate(self.get_random_deg(), expand=True)

        bg=bg.resize(mask_text.size)
        bg.paste(mask_text, mask_text)

        bg = self.img_transforms(bg)
        
        pixel_values = processor(bg, return_tensors="pt").pixel_values
        tokenized_text = tokenizer(text, return_tensors="pt", padding='max_length', max_length=self.max_length)
        input_ids = tokenized_text.input_ids[0]
        attention_mask = tokenized_text.attention_mask[0]
        
        output = {
          'pixel_values':pixel_values, 
          'input_ids':input_ids,
          'attention_mask':attention_mask
        }
      except:
        continue
      break
    
    return output

dataGenerator = DataGenerator()

# torch.save(dataGenerator, 'dataGenerator.pt')

# dataGenerator = torch.load('dataGenerator.pt')


data = {
    'pixel_values':[],
    'input_ids':[],
    'attention_mask':[]
}

num_images = 100000
for i, x in tqdm(enumerate(dataGenerator), total=num_images):
    data['pixel_values'].append(x['pixel_values'])
    data['input_ids'].append(x['input_ids'])
    data['attention_mask'].append(x['attention_mask'])
    if i==num_images:
        break
   
torch.save(data, 'data.pt')
