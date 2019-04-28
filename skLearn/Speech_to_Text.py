#import library
import cmu_sphinx4

#read audio file
audio_URL=''    #file path

#transcribe
transcriber=cmu_sphinx4.Transcriber(audio_URL)

#Print it out
for line in transcriber.transcript_stream():
    print line