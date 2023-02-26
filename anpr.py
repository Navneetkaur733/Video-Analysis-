import requests
import cv2

BASE_URL = 'https://api.platerecognizer.com/v1/plate-reader/'
API_KEY = '956d5a568bd27725879ba8741c8150950989a7cf'

regions = ['mx', 'us-ca', 'in'] 

def read_plates(img):
    image_bytes = cv2.imencode('.png', img)[1].tobytes()
    response = requests.post(BASE_URL, data=dict(regions=regions), files=dict(upload=image_bytes), headers={'Authorization': f'Token {API_KEY}'})
    status = response.status_code
    if not (status >= 200 and status < 300): return []
    return response.json().get('results', [])
    
    
if __name__ == '__main__':
    img_path = 'res/imgs/license_plates/images/Cars0.png'
    img = cv2.imread(img_path)
    res = read_plates(img)

    from pprint import pprint
    pprint(res)
