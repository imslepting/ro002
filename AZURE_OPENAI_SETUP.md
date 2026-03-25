# Azure OpenAI 配置指南

## 快速設置

### 方式 1：配置文件 + 環境變數（推薦）

1. **編輯 `config/settings.yaml`**
   ```yaml
   vlm:
     provider: "openai"
     azure_openai:
       enabled: true
       endpoint: "https://your-resource.openai.azure.com/"  # Azure Portal 中的 Endpoint
       deployment: "gpt-4o"  # 你的部署名稱（注意不是模型名，而是部署名）
       api_version: "2024-12-01-preview"
   ```

2. **設定環境變數**
   ```bash
   export OPENAI_API_KEY="your-azure-api-key"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
   ```

3. **運行程式**
   ```bash
   conda run -n ro002 python phase5_vlm_planning/test_vlm_agent.py
   ```

### 方式 2：只用環境變數（無需修改配置文件）

如果不想改配置文件，直接設定環境變數：
```bash
export OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
conda run -n ro002 python phase5_vlm_planning/test_vlm_agent.py
```

## 如何獲取 Azure 信息

1. 登入 [Azure Portal](https://portal.azure.com)
2. 找到你的 OpenAI 資源
3. 在左側菜單開啟 **Keys and Endpoint**
4. 複製：
   - **Endpoint** （如 `https://xxx.openai.azure.com/`）
   - **API Key** （任一個都可以）
   - 在 **Deployments** 選項卡查看部署名稱

## 部署名稱說明

- **Azure 部署名稱** ≠ 模型名稱
- 在 Azure Portal → OpenAI 資源 → **Deployments** 中查看
- 例如：你可能部署的是 `gpt-4o` 模型，但部署名稱可能是 `my-gpt4o-deployment`
- 配置文件中 `deployment` 欄位需填部署名稱

## API 版本

預設使用 `2024-12-01-preview`，支持 tool-use（agent 所需）。

## 故障排查

**錯誤：`Incorrect API key provided`**
- 檢查 API key 是否複製完整（無空格）
- 確認 key 未過期
- 確認 Azure 訂閱仍有效

**錯誤：`404 Not Found - Invalid deployment ID`**
- 檢查 `AZURE_OPENAI_DEPLOYMENT` 或配置文件中的 `deployment` 是否正確
- 確認部署在該資源中存在

**錯誤：`InvalidUrl`**
- Endpoint 應以 `/` 結尾：`https://your-resource.openai.azure.com/`

## 測試連接

```bash
conda run -n ro002 python -c "
import os
from phase5_vlm_planning.src.vlm_client import create_vlm

vlm = create_vlm('openai',
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.environ.get('AZURE_OPENAI_DEPLOYMENT')
)
print('Azure OpenAI 連接成功！')
"
```
