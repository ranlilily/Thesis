## Convert melodies into symbol sequencees
## [C - r D] = ['C-2', 'r-1', 'D-1']
## Output = all melodies, as a list of lists

import music21 as m21
import statistics as stat

# ----- 保留符合特定拍子記號的樂譜 -----
def has_required_time_signature(ts, time_signature):   #time_signature 拍子記號（4/4 拍等）
    if ts.ratioString != time_signature:
        return False
    return True


# ----- 保留符合特定延長音的樂譜 -----
def has_acceptable_durations(thepart, acceptable_durations):  # 檢查樂譜中的音符是否具有可接受的時值（duration）
    # as long as one false, return false
    for note in thepart.flat.notesAndRests:
      if note.duration.quarterLength not in acceptable_durations:
         return False
    return True


# ----- 從 m21 套件中取得 Data (樂譜) -----  
def extract_melody(song, voice):
  melody = 0  # in case no soprano
  parts = song.getElementsByClass(m21.stream.Part)   # 這行程式碼從給定的音樂物件 song 中獲取所有的樂器聲部
  for i in range(0, len(parts)):
    if voice in parts[i].partName:
      melody = parts[i]
      break      # 找到目標聲部後，終止迴圈，不再繼續查找其他聲部
  return melody


# ----- 將音樂轉調到指定的 C 大調或 A 小調 -----  
def transpose_key(thepart, key):     
    ##- transpose to C/Am key
    if key.mode == "major":
       interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
       interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A")) 
    thepart_trans = thepart.transpose(interval)
    return thepart_trans

# 在轉調音樂時，可以使用 m21.interval.Interval 來計算原始音樂和目標調性之間的音程，然後將音樂中的每個音高根據計算得到的音程進行移動，從而達到轉調的目的。


# ----- 調整音高 -----  
def transpose_octave(thepart):  
    pitch_seq = [x.pitch.midi for x in thepart.flat.notes]
    pitch_avg = stat.mean(pitch_seq)
    if pitch_avg > 71:
       thepart_trans = thepart.transpose(-12)
    elif pitch_avg < 60:
       thepart_trans = thepart.transpose(12)
    else:
       thepart_trans = thepart
    return thepart_trans

# ?然莉問題: 是否可以不調整、調整後是否可以辨識回來


# ----- 編碼成符號序列 -----
def encode_sequence_char(thepart):  
    encoded_seq = []
    # Combine the pitch and duration into a single character
    # [C - r D] = ['C-2', 'r-1', 'D-1']
    for event in thepart.flat.notesAndRests:      
      event_dur = event.duration.quarterLength
      #  handle notes
      if isinstance(event, m21.note.Note):
        event_pitch = event.pitch
      # handle rests
      if isinstance(event, m21.note.Rest):
        event_pitch = 'R'
      # make symbols
      symbol = f"{event_pitch}-{event_dur}"
      encoded_seq.append(symbol)
    return encoded_seq






def make_melody_symbol_sequences(songs, time_signature, acceptable_durations):
  
  ## (i) extract soprano
  ## (ii) filter out songs - TS not 4/4
  ## (iii) filter out melodies of non-acceptable durations
  ## (iv) transpose to C/Am key and to C4-B4 range
  ## (v) encode melody to symbol sequence then merge all
 
  encoded_seqs = []
   
  for i, song in enumerate(songs):      

    song = songs[i].parse()
    parts = song.getElementsByClass(m21.stream.Part)

    ## (i) extract soprano
    melody = extract_melody(song, "Soprano")
    if melody == 0:
       continue

    ## (ii) filter out songs not 4/4
    ts = parts[0].getTimeSignatures()[0]
    if not has_required_time_signature(ts, time_signature):
        continue

    ## (iii) filter out melodies of non-acceptable durations
    if not has_acceptable_durations(melody, acceptable_durations):
        continue

    ## (iv) transpose to C/Am key and to C4-B4 range
    key = song.analyze("key")
    melody = transpose_key(melody, key)
    melody = transpose_octave(melody) 

    ## (v) encode melody to symbol sequence then merge them
    encoded_seq = encode_sequence_char(melody)
    encoded_seqs.append(encoded_seq)

  return encoded_seqs




'''
1. 提取女高音旋律：
程式碼首先從歌曲中提取女高音旋律，這是由 extract_melody 函式完成的。如果找不到女高音部分，則跳過當前歌曲。
篩選4/4拍：

2. 接下來，程式碼檢查歌曲的拍子是否為4/4拍，只有符合此要求的歌曲才會被保留。

3. 篩選指定節奏長度：此步驟過濾掉旋律部分中不符合指定節奏長度的音符。只有符合要求的節奏長度才會被保留。

4. 調整音高及音域：歌曲的調性被分析後，女高音旋律部分會被調整到C大調或A小調，並在C4到B4的音域範圍內進行調整。

5. 編碼成符號序列：最後，程式碼將調整後的女高音旋律部分編碼成符號序列，這是由 encode_sequence_char 函式完成的。編碼後的符號序列被添加到 encoded_seqs 列表中。
返回編碼後的符號序列列表：

6. 最終，函式返回 encoded_seqs 列表，其中包含了所有歌曲的女高音旋律部分的符號序列。
這個函式主要用於準備將歌曲數據集輸入到模型中進行訓練，這些符號序列將作為模型的輸入特徵。
'''


