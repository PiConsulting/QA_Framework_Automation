import requests

url = "https://ca-zxkvhdr7ju7ku-frontend.graybay-13f3c191.eastus.azurecontainerapps.io/api/chat"
payload = {"input": "¿Cuál es la capital de Argentina?"}
headers = {"Content-Type": "application/json"}

response = requests.get(url, json=payload, headers=headers)
print(response.status_code)
print(response.text)
