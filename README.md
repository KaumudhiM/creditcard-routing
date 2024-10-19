# Instructions

First install all packages 

```
pip install -r requirements.txt
```

To pre-process data 
```
python src/data_processing.py
```

To create models 
```
python src/modeling.py
```

To run flask app 
```
python src/deployment.py
```

Use curl to test
```
curl -X POST -H "Content-Type: application/json" -d '{"country": "Germany", "amount": 100, "PSP": "Moneycard", "3D_secured": 1, "card": "Visa"}' http://127.0.0.1:5000/predict
```
