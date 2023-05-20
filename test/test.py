import unittest
import requests
import PIL.Image
import io
from requests import request

class MyTestCase(unittest.TestCase):
    def test_200(self):
        response = requests.request('GET', 'http://localhost:1791/')
        sample = response.content.decode()
        self.assertEqual(sample, 'Home page')  # add assertion here

    def test_imagenet(self):
        img = PIL.Image.open('../data/dog.jpg')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')

        with buffer as buf:
            buffer.seek(0)
            response = request('POST', 'http://localhost:1791/classify', data=buf)

        out = response.content.decode('utf-8')
        expected = 'Пембрук'
        
        self.assertIn(expected, out)


    def test_binary_classify(self):
        img = PIL.Image.open('../data/dog.jpg')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')

        with buffer as buf:
            buffer.seek(0)
            response = request('POST', 'http://localhost:1791/classify/binary', data=buf)

        out = response.content.decode('utf-8')
        print(out)
        expected = 'Cat'
        self.assertEqual(expected, out)
if __name__ == '__main__':
    unittest.main()
