import torch
from pathlib import Path
from functools import partial
import asyncio
import orjson

# Importações principais do Delphi
from delphi.config import RunConfig, CacheConfig, ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset
from delphi.clients import Offline # Para rodar o LLM avaliador localmente com vLLM
from delphi.explainers import DefaultExplainer
from delphi.scorers import DetectionScorer, FuzzingScorer
from delphi.pipeline import Pipeline, Pipe, process_wrapper
from transformers import AutoTokenizer

# --- 1. Configuração Geral ---
# Esta configuração agora irá controlar o processo de interpretação.
model_name = "EleutherAI/pythia-70m-deduped"
hookpoints = ['layers.2']
latents_path = Path("./results/my_run/latents") # Onde seu cache foi salvo
explanations_path = Path("./results/my_run/explanations")
scores_path = Path("./results/my_run/scores")

# Crie os diretórios de saída
explanations_path.mkdir(parents=True, exist_ok=True)
scores_path.mkdir(parents=True, exist_ok=True)

# Configuração para carregar os dados do cache e amostrar exemplos
constructor_cfg = ConstructorConfig(
    min_examples=20, # Mínimo de ativações para uma feature ser explicada
    example_ctx_len=128 # Comprimento dos exemplos mostrados ao LLM
)
sampler_cfg = SamplerConfig(
    n_examples_train=20, # Nº de exemplos para gerar a explicação
    n_examples_test=30  # Nº de exemplos para avaliar a explicação
)

# --- 2. Inicializar o Cliente LLM para Interpretação ---
# Escolha o LLM que fará as explicações e avaliações.
# Lembre-se que ele precisa de VRAM suficiente. Llama-3 8B é uma boa opção.
explainer_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

llm_client = Offline(
    model=explainer_model_name,
    max_memory=0.85, # Utilização da VRAM da GPU
    max_model_len=4096,
    num_gpus=1
)

# É necessário um tokenizer para o modelo base para decodificar os tokens dos exemplos
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- 3. Carregar o Cache com LatentDataset ---
# Esta classe carrega os dados dos arquivos .safetensors de forma lazy (sob demanda)
print("Carregando o cache de ativações...")
dataset = LatentDataset(
    raw_dir=latents_path,
    sampler_cfg=sampler_cfg,
    constructor_cfg=constructor_cfg,
    modules=hookpoints,
    tokenizer=tokenizer
)

# --- 4. Definir a Pipeline de Explicação e Avaliação ---

# Função para salvar a explicação em um arquivo
def explainer_postprocess(result):
    # O nome do arquivo será baseado na camada e no índice da feature
    explanation_file = explanations_path / f"{result.record.latent}.json"
    with open(explanation_file, "wb") as f:
        f.write(orjson.dumps(result.explanation))
    return result

# Função para preparar os dados para os scorers
def scorer_preprocess(result):
    record = result.record
    record.explanation = result.explanation # Adiciona a explicação gerada ao registro
    record.extra_examples = record.not_active
    return record

# Função para salvar os resultados dos scorers
def scorer_postprocess(result, score_dir):
    score_file = score_dir / f"{result.record.latent}.json"
    with open(score_file, "wb") as f:
        f.write(orjson.dumps(result.score))
    return result

# Crie o "explicador"
explainer = DefaultExplainer(
    llm_client,
    verbose=True,
    threshold=0.3
)

# Crie os "avaliadores" (scorers)
fuzz_scorer = FuzzingScorer(llm_client, verbose=True, n_examples_shown=5)
detection_scorer = DetectionScorer(llm_client, verbose=True, n_examples_shown=5)

# Monte a pipeline
# A pipeline irá:
# 1. Pegar um item do `dataset` (dados de uma feature).
# 2. Passar para o `explainer_pipe` para gerar e salvar a explicação.
# 3. Passar o resultado para o `scorer_pipe` para avaliar e salvar os scores.
pipeline = Pipeline(
    dataset,
    # Etapa 1: Gerar Explicações
    Pipe(process_wrapper(explainer, postprocess=explainer_postprocess)),
    
    # Etapa 2: Avaliar Explicações
    Pipe(
        process_wrapper(
            fuzz_scorer,
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=scores_path / "fuzz")
        ),
        process_wrapper(
            detection_scorer,
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=scores_path / "detection")
        )
    )
)

# --- 5. Executar a Pipeline ---
async def main():
    print("Iniciando a pipeline de interpretação...")
    # max_concurrent define quantas features serão processadas em paralelo
    await pipeline.run(max_concurrent=10) 
    print("Interpretação concluída!")
    # Lembre-se de desligar o cliente vLLM para liberar a VRAM
    await llm_client.close()

if __name__ == "__main__":
    asyncio.run(main())

