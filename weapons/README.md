pip install  "numpy==1.26.3"
pip install --upgrade  "numexpr>=2.8.4"  "bottleneck>=1.3.6" "prometheus_client"


```bash
curl -X 'POST' \
  'http://localhost:9001/model/add' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_id": "weapons1",
  "model_type": "object-detection"
}'
```