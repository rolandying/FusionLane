import os
import shutil
import random
import time
import sys

from captcha.image import ImageCaptcha

CHAR_SET = ['0','1','2','3','4','5','6','7','8','9']

CHAR_SET_LEN = 10

CAPTCHA_LEN = 4

CAPTCHA_IMAGE_PATH = '/home/roland/captcha/images/'
TEST_IMAGE_PATH = '/home/roland/captcha/test/'

TEST_IMAGE_NUMBER = 50


def generate_captcha_image(charSet = CHAR_SET, charSetLen=CHAR_SET_LEN, captchaImgPath=CAPTCHA_IMAGE_PATH):   
    k  = 0
    total = 1
    for i in range(CAPTCHA_LEN):
        total *= charSetLen
        
    for i in range(charSetLen):
        for j in range(charSetLen):
            for m in range(charSetLen):
                for n in range(charSetLen):
                    captcha_text = charSet[i] + charSet[j] + charSet[m] + charSet[n]
                    image = ImageCaptcha()
                    image.write(captcha_text, captchaImgPath + captcha_text + '.jpg')
                    k += 1
                    sys.stdout.write("\rCreating %d/%d" % (k, total))
                    sys.stdout.flush()
                                      
def prepare_test_set():
    fileNameList = []    
    for filePath in os.listdir(CAPTCHA_IMAGE_PATH):
        captcha_name = filePath.split('/')[-1]
        fileNameList.append(captcha_name)
    random.seed(time.time())
    random.shuffle(fileNameList) 
    for i in range(TEST_IMAGE_NUMBER):
        name = fileNameList[i]
        shutil.move(CAPTCHA_IMAGE_PATH + name, TEST_IMAGE_PATH + name)
                        
if __name__ == '__main__':
    generate_captcha_image(CHAR_SET, CHAR_SET_LEN, CAPTCHA_IMAGE_PATH)
    prepare_test_set()
    sys.stdout.write("\nFinished")
    sys.stdout.flush()  
