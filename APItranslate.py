#simple trial to translate using googletrans API
import json



from googletrans import Translator


#opens English File
file = "original.en"

#file2 = what to save as
file2 = open("machinetranslated.txt", mode= "w", encoding='utf-8')
with open(file, encoding='utf-8') as f:
	lines = f.readlines()

counter = 0

#prints translated text only
for line in lines:
	translator = Translator()
	try:
		translation = translator.translate(line, dest='nl')
		counter +=1
		file2.write(translation.text)
		file2.write("\n")
	except Exception as e:
		print(str(e))
		continue
		
	

file.close()
file2.close()
