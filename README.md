# Awesome-LLM4Security 
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)  
> 一个关于网络安全模型的精选资源列表，包含模型/项目、论文、数据以及相关产品。

## 目录
- [简介](#简介)
- [资源列表](#资源列表)
  - [项目](#项目)
  - [论文](#论文)
  - [数据集](#数据集)
  - [相关产品](#相关产品)
- [贡献](#贡献)
- [附录](#附录)
  - [许可证](#许可证)
  - [Stars History](#点赞历史)  


## 简介
这是一个精心整理的网络安全模型资源汇总，旨在为研究人员、工程师及安全爱好者提供一个全面的参考集合。本项目覆盖了模型/项目、学术论文、数据集以及相关产品信息，帮助你深入了解和应用网络安全领域的最新进展。 

## 资源列表

### 项目
| 名称                                            | 简介                                                         | 链接                                                         |
| ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SecureBERT: CyberSecurity Language Model        | 专门为网络安全任务量身定制的专用语言模型。该存储库记录了 SecureBERT 在使用操作码序列对恶意软件样本进行分类方面的研究、开发和实施。 | [SecureBert_Malware-Classification](https://github.com/kaushik-42/SecureBert_Malware-Classification) |
| SecureBERT-NER                                  | 该模型用于从安全咨询文本中提取命名实体。专门针对网络安全文本进行训练的命名实体识别 (NER) 模型。它可以识别身份、安全团队、工具、时间、攻击等各种实体。 | [cybersecurity-ner](https://github.com/PriyankaMohan94/cybersecurity-ner) |
| ceg-afpm                                        | 用于使用LLM句子分类进行安全漏洞误报识别                      | [ceg-afpm](https://github.com/arigig/ceg-afpm)               |
| CVE2TTP                                         | 一种从网络安全文本资源中评估、注释和提取威胁信息的自动化方法，以在 SecureBERT 之上开发模型，将 CVE 分类为 TTP。 | [CVE2TTP](https://github.com/ehsanaghaei/CVE2TTP)            |
| SecureBERT                                      | 一种用于表示网络安全文本数据的特定领域语言模型。             | [SecureBERT](https://github.com/ehsanaghaei/SecureBERT)<br />[SecureBERT-plus](https://github.com/ehsanaghaei/SecureBERT-plus)<br />[SecureDeBERTa](https://github.com/ehsanaghaei/SecureDeBERTa)<br />[SecureGPT](https://github.com/ehsanaghaei/SecureGPT) |
| Finetuning_SecurityLLM                          | 基于BERT微调的网络威胁检测系统                               | [Finetuning_SecurityLLM](https://github.com/GeorgeNhj/Finetuning_SecurityLLM) |
| flipkart_project                                | 使用大型语言模型进行自动合规性监控（LLMs）”项目，利用 BERT 和 RoBERTa 等尖端LLMs技术的力量，彻底改变合规性监控和实施 | [flipkart_project](https://github.com/Gauriiikaaa/flipkart_project) |
| PentestGPT                                      | 支持 GPT 的渗透测试工具                                      | [PentestGPT](https://github.com/GreyDGL/PentestGPT)          |
| WhiteRabbitNeo                                  | 一个可用于进攻和防御网络安全的模型系列【包含7B、13B、33B】   | [WhiteRabbitNeo](https://huggingface.co/WhiteRabbitNeo)      |
| Whiterabbitneo-Pentestgpt                       | 将专门针对网络安全的开源模型whiterabbitneo和GPT 4的提示技术PentestGPT结合起来，让GPT 4进入hack the box用户的前1%，打造一个完全开放的渗透测试的源解决方案。 | [Whiterabbitneo-Pentestgpt](https://github.com/isamu-isozaki/whiterabbitneo-pentestgpt) |
| LATTE                                           | 结合LLM和程序分析的二进制污点分析引擎。结合LLMs来实现自动化的二进制污点分析。这克服了传统污点分析需要手动定制污点传播规则和漏洞检查规则的局限性 | [LATTE](https://github.com/puzhuoliu/LATTE)                  |
| AVScan2Vec                                      | 一种序列到序列自动编码器，可以将恶意文件的防病毒结果嵌入到向量中。然后，这些向量可用于下游 ML 任务，例如分类、聚类和最近邻查找。【将防病毒扫描报告转换为向量表示，有效处理大规模恶意软件数据集，并在恶意软件分类、聚类和最近邻搜索等任务中表现良好】 | [AVScan2Vec](https://github.com/boozallen/AVScan2Vec)        |
| PassGPT                                         | 一个针对密码泄露从头开始训练的 GPT-2 模型。利用 LLMs 的密码生成模型，引入了引导式密码生成，其中 PassGPT 的采样过程生成符合用户定义约束的密码。这种方法通过生成更多以前未见过的密码，优于利用生成对抗网络（GAN）的现有方法，从而证明了LLMs在改进现有密码强度估计器方面的有效性 | [PassGPT](https://github.com/javirandor/passgpt)             |
| pwned-by-passgpt                                | 使用 Have I Been Pwned (HIBP) 数据集进行密码破解研究，以评估 PassGPT 大型语言模型 (LLM) 的有效性。 | [pwned-by-passgpt](https://github.com/sean-t-smith/pwned-by-passgpt) |
|                                                 |                                                              |                                                              |
| LLM Security 101                                | 深入LLM安全领域：进攻和防御工具的探索，揭示它们目前的能力。  | [LLM Security 101](https://github.com/Seezo-io/llm-security-101) |
| SecurityGPT                                     | 使用大型语言模型 (LLMs) 增强软件源代码中的安全错误报告 (SBR) 的分类。我们开发并微调了 LLMs 来解决识别代码中安全漏洞的关键任务。 | [SecurityGPT](https://github.com/Alexyskoutnev/SecurityGPT)  |
| ChatCVE                                         | 帮助组织分类和聚合 CVE（常见漏洞和暴露）信息。通过利用最先进的自然语言处理，ChatCVE 使每个人都可以访问详细的软件物料清单 (SBOM) 数据 | [ChatCVE](https://github.com/jasona7/ChatCVE)                |
| SecGPT                                          | SecGPT的目标是结合LLM，对网络安全进行更多贡献，包括渗透测试、红蓝对抗、CTF比赛和其他方面。汇总现有的插件功能，并通过AI进行决策。基于这些决策，它构建基础行为逻辑。然后，根据此逻辑，它调用本地插件功能，尝试完成网站渗透、漏洞扫描、代码审计和报告撰写等任务 | [SecGPT](https://github.com/ZacharyZcR/SecGPT)               |
| SecGPT-云起无垠                                 | 网络安全大模型。探索各种网络安全任务，漏洞分析、溯源分析、流量分析、攻击研判、命令解释、网安知识问答、高质量网络安全训练集、DPO强化学习 | [secgpt](https://github.com/Clouditera/secgpt)               |
| HackerGPT                                       | 用于网络应用程序黑客攻击的值得信赖的道德黑客LLM，针对网络和网络黑客攻击，使用的开源黑客工具进行黑客攻击。 | [HackerGPT]()                                                |
| AutoAudit                                       | AutoAudit作为专门针对网络安全领域的大语言模型，其目标是为安全审计和网络防御提供强大的自然语言处理能力。它具备分析恶意代码、检测网络攻击、预测安全漏洞等功能，为安全专业人员提供有力的支持。 | [AutoAudit](https://github.com/ddzipp/AutoAudit)             |
| Agentic LLM                                     | 开源 Agentic LLM 漏洞扫描程序                                | [Agentic LLM](https://github.com/msoedov/agentic_security)   |
| Garak                                           | LLM 漏洞扫描器                                               | [Garak](https://github.com/leondz/garak)                     |
| SourceGPT                                       | 构建在 ChatGPT 之上的源代码分析器和提示管理器（可做代码扫描） | [SourceGPT]()                                                |
| ChatGPTScan                                     | 由 ChatGPT 提供支持的代码扫描                                | [ChatGPTScan](https://github.com/YulinSec/ChatGPTScanner)    |
| ChatGPT Code Analyzer                           | 利用ChatGPT 进行的代码分析器                                 | [chatgpt-code-analyzer](https://github.com/MilindPurswani/chatgpt-code-analyzer) |
| Hacker AI                                       | 检测源代码中的漏洞的在线工具（闭源-公司）                    | [hacker-ai](https://hacker-ai.ai/#hacker-ai)                 |
| GPTLens                                         | 基于LLM的智能合约漏洞检测                                    | [GPTLens](https://github.com/AvijeetRanawat/GPTLens)         |
| Audit GPT                                       | 微调 GPT 以进行智能合约审计                                  | [Audit GPT](https://github.com/fuzzland/audit_gpt)           |
| VulChatGPT                                      | 使用 IDA PRO HexRays 反编译器和 OpenAI(ChatGPT) 来查找二进制文件中可能存在的漏洞 | [VulChatGPT](https://github.com/ke0z/vulchatgpt)             |
| Ret2GPT                                         | 利用 OpenAI API 的能力，RET2GPT 可以为二进制文件提供全面而详细的分析，使其成为 CTF Pwners 不可或缺的工具。 | [Ret2GPT](https://github.com/DDizzzy79/Ret2GPT)              |
| LLM-CodeSecurityReviewer                        | 使用 Ollama(LLM) 检查代码是否存在潜在的不良行为              | [LLM-CodeSecurityReviewer](https://github.com/t41372/LLM-CodeSecurityReviewer) |
| LLM-SOC                                         | 基于大语言模型的安全运营辅助增强工具，**RAG框架**（**目前只有README**） | [LLM-SOC](https://github.com/404notf0und/LLM-SOC)            |
| RagSecOps                                       | LLMs + **RAG** + CVEs + Security = SecAIOps                  | [RagSecOps](https://github.com/rcarrat-AI/ragsecops)         |
| FlipLogGPT                                      | 使用向量存储进行日志和安全分析的交互式LLM（**RAG框架**）     | [FlipLogGPT](https://github.com/adarshpalaskar1/FlipLogGPT_LLM) |
| pentestpal                                      | 不断发展的LLM驱动的工具，以协助渗透测试人员和安全研究人员（**RAG框架**） | [pentestpal](https://github.com/marklechner/pentestpal)      |
| Sovereign Chat                                  | FOSS AI 聊天机器人可以回答有关在线隐私和安全的所有问题（**RAG框架**）<br />[Chainlit](https://docs.chainlit.io/get-started/overview) + [Embedchain](https://github.com/embedchain/embedchain/tree/main) + [Ollama](https://ollama.com/) | [Sovereign Chat](https://github.com/Marconius-Solidus/Sovereign-Chat) |
| Q-A-bot                                         | 利用大型语言模型 （LLM） 功能与网络安全文档交互的个性化机器人。（**RAG框架**） | [Q-A-bot](https://github.com/sheshiisree/Q-A-bot)            |
| ZenGuard AI                                     | 将生产级、低代码 LLM（大型语言模型）护栏集成到其生成式 AI 应用程序中。【及时注入检测、越狱检测、个人身份信息检测、关键字检测等】 | [fast-llm-security-guardrails](https://github.com/ZenGuard-AI/fast-llm-security-guardrails) |
| cyber-security-llm-agents                       | 使用大型语言模型 (LLMs) 执行网络安全日常工作中常见任务的代理集合 | [cyber-security-llm-agents](https://github.com/NVISOsecurity/cyber-security-llm-agents) |
| Galah                                           | 一个 LLM（大型语言模型）驱动的 Web 蜜罐，目前与 OpenAI API 兼容，能够模仿各种应用程序和动态响应任意 HTTP 请求。（GO语言） | [Galah](https://github.com/0x4D31/galah)                     |
| CyberSecurityLLMTest                            | 测试（数据/提示）大型语言模型是否具有作为网络安全专家执行的能力 | [CyberSecurityLLMTest](https://github.com/ybdesire/CyberSecurityLLMTest) |
| OpenAI and FastAPI - Text summarization         | 一个基于 OpenAI 的 GPT-3.5 和 GPT-4 API 生成威胁情报摘要报告的工具 | [OpenAI and FastAPI - Text summarization](https://github.com/EC-DIGIT-CSIRC/openai-cti-summarizer) |
| Security LLaMA2 Fine-tuning                     | 利用LLama2进行微调安全领域                                   | [Security LLaMA2 Fine-tuning](https://github.com/KaitaoQiu/security_llm) |
| LLM-security                                    | ✨✨                                                           | [LLM-security](https://github.com/Anonymous1234343/LLM-security) |
| LLM_Security                                    | 利用RAG与ChatGPT结合实现LLM Security                         | [LLM_Security](https://github.com/BoB-Dev-Top30/LLM_Security) |
| LLM Security Chatbot                            | LLM 安全聊天机器人旨在帮助理解和研究网络安全研究。主要是 POC。该聊天机器人使用 Mistral 7B v1 构建，并使用 Streamlit 集成到用户友好的界面中，利用自然语言处理为广泛的安全问题提供深入的分析和潜在的缓解策略 | [LLM Security Chatbot](https://github.com/jwalker/llm_security_chatbot) |
| smartgrid-llm                                   | 在智能电网中实践大型语言模型的风险：威胁建模和验证           | [smartgrid-llm](https://github.com/jiangnan3/smartgrid-llm)  |
| AISploit                                        | AISploit 是一个 Python 包，旨在支持红队和渗透测试人员利用大型语言模型人工智能解决方案。它提供工具和实用程序来自动执行与基于人工智能的安全测试相关的任务。 | [AISploit ](https://github.com/hupe1980/aisploit)            |
| PyRIT                                           | 用于生成 AI 的 Python 风险识别工具 (PyRIT) 是一个开放访问自动化框架，使安全专业人员和机器学习工程师能够使用红队基础模型及其应用程序。 | [PyRIT](https://github.com/Azure/PyRIT)                      |
| Experiment AI Nutrition-Pro                     | 用于威胁建模和安全审查以及使用 OpenAI GPT-4 的研究项目       | [Experiment AI Nutrition-Pro](https://github.com/xvnpw/ai-nutrition-pro-design-gpt4) |
| SecurityGuardianAI                              | SecurityGuardianAI 是一款主动式云安全分析应用程序，旨在帮助识别云基础设施中的潜在安全威胁和漏洞。应用程序应该能够提供实时监控、分析和报告，以跟踪云服务器上可能发生的任何恶意活动【**实际没有用LLMs**】 | [SecurityGuardianAI](https://github.com/vps/SecurityGuardianAI) |
| Admyral                                         | 一款开源网络安全自动化和调查助手。网络安全自动化和调查助理   | [Admyral](https://github.com/Admyral-Security/admyral)       |
| Real-Time-Network-Traffic-Analysis-with-LLM-API | 使用大型语言模型 （LLM） API 进行实时网络流量分析。通过对 DNS 查询进行实时分析和分类，探索大型语言模型 （LLMs） 在网络安全方面的潜力 | [Real-Time-Network-Traffic-Analysis-with-LLM-API](https://github.com/FerdinandPaul/Real-Time-Network-Traffic-Analysis-with-LLM-API) |
| CVE2ATT-CK-LLM                                  | 一种利用LLM（大型语言模型）功能自动将 CVE 描述映射到 ATT&CK 技术的工具。旨在通过弥合漏洞和对抗性策略之间的差距来增强威胁情报和安全意识。 | [CVE2ATT-CK-LLM](https://github.com/vkeilo/CVE-2-ATT-CK-LLM) |
| MitreTagging                                    | 开发 MITRE ATT&CK 标记模型的项目，该模型采用安全发现、描述和分析，并使用适当的 ATT&CK 策略和技术对其进行标记 | [MitreTagging](https://github.com/Lifebloom-AI/MitreTagging) |
| CodeScanGPT                                     | 基于 GPT 和 OpenAI API 构建的实验性静态应用程序安全测试 （SAST） 扫描程序。 | [CodeScanGPT](https://github.com/chasepd/CodeScanGPT)        |
|                                                 |                                                              |                                                              |
|                                                 |                                                              |                                                              |


### 论文

### 数据集

### 相关产品



## 贡献
欢迎为这个列表做出贡献！你可以通过提交一个pull request来添加、修改或删除资源。在提交之前，请确保你遵循了以下准则：
- 确保资源的质量上乘，并且与网络安全模型的主题相关。
- 在添加资源时，请按照相应的分类进行排序，并保持列表的整洁。
- 提供资源的清晰描述，包括标题、链接以及简短的介绍。
如果你有任何疑问或建议，请随时通过GitHub的issue与我联系。

## 附录
### 许可证
本项目遵循 [MIT](LICENSE) 许可证

### 点赞历史
[![Star History Chart](https://api.star-history.com/svg?repos=liu673/Awesome-LLM4Security&type=Date)](https://star-history.com/#liu673/Awesome-LLM4Security&Date)


