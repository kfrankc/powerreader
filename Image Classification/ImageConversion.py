import pdf2image
import os
import win32com.client

pptPath = f'{os.getcwd()}\\Slides.pptx'
pdfPath = f'{os.getcwd()}\\Image Classification\\Slides'
jpgPath = f'{os.getcwd()}\\Image Classification\\Original Slides'
try:
    print(pptPath)
    powerpoint = win32com.client.Dispatch("Powerpoint.Application")
    deck = powerpoint.Presentations.Open(f'{pptPath}')
    deck.SaveAs(pdfPath, 32) # formatType = 32 for ppt to pdf
    deck.Close()
    powerpoint.Quit()
    print('done')
    # pass
except:
    print('could not open')

images = pdf2image.convert_from_path(f'{pdfPath}.pdf')
for index, image in enumerate(images):
    image.save(f'{jpgPath}\\Slide {index}.jpg')