import unittest
import requests
from api import remove_stopwords

class ApiTest(unittest.TestCase):
    API_URLs = ["http://127.0.0.1:8000/glove", "http://127.0.0.1:8000/w2v"]
    
    def test_api(self):
        
        text_list = ["گرون بود",
                     "راضی کننده و عالی بود",
                     "بس کنید دیگه",
                     "کاش قیمت های بسته‌های جدید رو بگذارید",
                     "رایتل",
                     "ابرانسل دزد کثیف",
                     1231.123123,
                     True,
                     12,
                     ]

        for api_url in self.API_URLs:           
            for item in text_list:
                r = requests.post(api_url, data = {"text":item})
                self.assertEqual(r.status_code, 200)
                neg_prob = str(r.content).split(",")[0].split(" : ")[1]
                self.assertGreaterEqual(float(neg_prob), 0)
                self.assertLessEqual(float(neg_prob), 1)

    def test_remove_stopwords(self):
        self.assertEqual(remove_stopwords("قیمت‌های با تخفیف"), "قیمت‌های تخفیف")

if __name__ == "__main__":
    unittest.main()