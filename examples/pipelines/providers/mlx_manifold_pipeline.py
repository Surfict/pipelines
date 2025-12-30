"""

title: MLX Manifold Pipeline

author: justinh-rahb

date: 2024-05-28

version: 2.1

license: MIT

description: A pipeline for generating text using Apple MLX Framework with dynamic model loading and reasoning support.

requirements: requests, mlx-lm, huggingface-hub, psutil

"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests
import subprocess
import logging
from huggingface_hub import login
import time
import psutil
import re

class Pipeline:
    class Valves(BaseModel):
        MLX_DEFAULT_MODEL: str = "mlx-community/Meta-Llama-3-8B-Instruct-8bit"
        MLX_MODEL_FILTER: str = "mlx-community"
        MLX_STOP: str = "<|start_header_id|>,<|end_header_id|>,<|eot_id|>"
        MLX_CHAT_TEMPLATE: str | None = None
        MLX_USE_DEFAULT_CHAT_TEMPLATE: bool | None = False
        HUGGINGFACE_TOKEN: str | None = None

    def __init__(self):
        # Pipeline identification
        self.type = "manifold"
        self.id = "mlx"
        self.name = "MLX/"

        # Initialize valves and update them
        self.valves = self.Valves()
        self.update_valves()

        # Server configuration
        self.host = "localhost"
        self.port = None

        # Model management
        self.models = self.get_mlx_models()
        self.current_model = None
        self.server_process = None
        
        # Timing tracking
        self.request_start_time = None

        # Start the MLX server with the default model
        self.start_mlx_server(self.valves.MLX_DEFAULT_MODEL)

    def update_valves(self):
        """Update pipeline configuration based on valve settings."""
        if self.valves.HUGGINGFACE_TOKEN:
            login(self.valves.HUGGINGFACE_TOKEN)
        self.stop_sequence = self.valves.MLX_STOP.split(",")

    def get_mlx_models(self):
        """Fetch available MLX models based on the specified pattern."""
        try:
            cmd = [
                'mlx_lm.manage',
                '--scan',
                '--pattern', self.valves.MLX_MODEL_FILTER,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            content_lines = [line for line in lines if line and not line.startswith('-')]
            models = []
            for line in content_lines[2:]:
                parts = line.split()
                if len(parts) >= 2:
                    repo_id = parts[0]
                    models.append({
                        "id": f"{repo_id.split('/')[-1].lower()}",
                        "name": repo_id
                    })

            if not models:
                models.append({
                    "id": f"mlx.{self.valves.MLX_DEFAULT_MODEL.split('/')[-1].lower()}",
                    "name": self.valves.MLX_DEFAULT_MODEL
                })

            return models

        except Exception as e:
            logging.error(f"Error fetching MLX models: {e}")
            return [{
                "id": f"mlx.{self.valves.MLX_DEFAULT_MODEL.split('/')[-1].lower()}",
                "name": self.valves.MLX_DEFAULT_MODEL
            }]

    def pipelines(self) -> List[dict]:
        """Return the list of available models as pipelines."""
        return self.models

    def start_mlx_server(self, model_name):
        """Start the MLX server with the specified model."""
        model_id = f"mlx.{model_name.split('/')[-1].lower()}"
        if self.current_model == model_id and self.server_process and self.server_process.poll() is None:
            logging.info(f"MLX server already running with model {model_name}")
            return

        self.stop_mlx_server()
        self.port = self.find_free_port()

        command = [
            "mlx_lm.server",
            "--model", model_name,
            "--port", str(self.port),
        ]

        if self.valves.MLX_CHAT_TEMPLATE:
            command.extend(["--chat-template", self.valves.MLX_CHAT_TEMPLATE])
        elif self.valves.MLX_USE_DEFAULT_CHAT_TEMPLATE:
            command.append("--use-default-chat-template")

        logging.info(f"Starting MLX server with command: {' '.join(command)}")
        self.server_process = subprocess.Popen(command)
        self.current_model = model_id
        logging.info(f"Started MLX server for model {model_name} on port {self.port}")

        self.wait_for_server_ready()

    def wait_for_server_ready(self, max_wait=120):
        """Wait for MLX server to be ready with health checks."""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
                if response.status_code == 200:
                    logging.info(f"MLX server ready after {time.time() - start_time:.1f}s")
                    return
            except:
                pass
            time.sleep(2)
        logging.warning(f"MLX server may not be ready after {max_wait}s")

    def stop_mlx_server(self):
        """Stop the currently running MLX server."""
        if self.server_process:
            try:
                process = psutil.Process(self.server_process.pid)
                for proc in process.children(recursive=True):
                    proc.terminate()
                process.terminate()
                process.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
            except psutil.TimeoutExpired:
                logging.warning("Timeout while terminating MLX server process")
            finally:
                self.server_process = None
                self.current_model = None
                self.port = None
                logging.info("Stopped MLX server")

    def find_free_port(self):
        """Find and return a free port to use for the MLX server."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def add_spaces_to_text(self, text: str) -> str:
        """Add proper spacing to compressed text."""
        # Add space after punctuation followed by a letter
        text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
        
        # Add space between lowercase and uppercase letters (camelCase)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Add space after closing quotes followed by letters
        text = re.sub(r'(["\'])([A-Za-z])', r'\1 \2', text)
        
        # Add space before opening quotes if preceded by letters
        text = re.sub(r'([A-Za-z])(["\'])', r'\1 \2', text)
        
        # Add space around common words that might be compressed
        text = re.sub(r'([a-z])(The|What|How|Why|When|Where|Who|I|You|We|They)', r'\1 \2', text)
        
        return text
    
    def clean_response(self, text: str) -> str:
        """Remove keepalive messages and convert thinking tokens to Open WebUI format."""
        # Remove keepalive messages
        text = re.sub(r'keepalive \d+/\d+:\s*', '', text)

        # Extract thinking and final answer
        # Matches [THINK]...[/THINK] with or without spaces
        think_pattern = r'\[THINK\](.*?)\[/THINK\]\s*(.*)'
        match = re.search(think_pattern, text, re.DOTALL)

        if match:
            thinking = match.group(1).strip()
            answer = match.group(2).strip()

            # Add spaces to compressed thinking text for readability
            thinking = self.add_spaces_to_text(thinking)
            
            # Also ensure answer has proper spacing
            answer = self.add_spaces_to_text(answer)

            # Convert to Open WebUI format: <think>...</think> followed by response
            # Open WebUI will automatically display <think> content in the thinking section
            return f"<think>{thinking}</think>\n\n{answer}"

        return text

    def process_stream(self, stream, start_time):
        """Process streaming responses to convert thinking tokens to Open WebUI format."""
        import json
        
        for line in stream:
            if line:
                decoded = line.decode('utf-8') if isinstance(line, bytes) else line
                
                # Skip keepalive messages
                if 'keepalive' in decoded.lower():
                    continue
                
                # Handle streaming JSON chunks
                if decoded.startswith('data: '):
                    try:
                        json_str = decoded[6:].strip()
                        if json_str and json_str != '[DONE]':
                            data = json.loads(json_str)
                            
                            # Handle usage statistics and add timing info
                            if 'usage' in data:
                                elapsed_time = time.time() - start_time
                                completion_tokens = data['usage'].get('completion_tokens', 0)
                                
                                # Calculate tokens per second
                                if elapsed_time > 0 and completion_tokens > 0:
                                    tokens_per_second = completion_tokens / elapsed_time
                                    # Add timing information for Open WebUI
                                    data['usage']['tokens_per_second'] = round(tokens_per_second, 2)
                                    data['usage']['generation_time'] = round(elapsed_time, 3)
                                
                                logging.info(f"Stream usage stats: {data['usage']}")
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    # Convert thinking markers to Open WebUI format
                                    content = content.replace('[THINK]', '<think>')
                                    content = content.replace('[/THINK]', '</think>')
                                    
                                    # Add spacing to compressed text
                                    content = self.add_spaces_to_text(content)
                                    
                                    # Update the delta with converted content
                                    delta['content'] = content
                                    data['choices'][0]['delta'] = delta
                                
                                # Yield the modified chunk
                                yield f"data: {json.dumps(data)}\n\n".encode('utf-8')
                            elif 'usage' in data:
                                # Usage-only chunk (usually final chunk)
                                yield f"data: {json.dumps(data)}\n\n".encode('utf-8')
                            else:
                                # Other chunks, pass through
                                yield line
                        else:
                            # [DONE] or empty, pass through
                            yield line
                    except json.JSONDecodeError:
                        yield line
                else:
                    yield line

    async def on_startup(self):
        """Perform any necessary startup operations."""
        logging.info(f"on_startup:{__name__}")

    async def on_shutdown(self):
        """Perform cleanup operations on shutdown."""
        self.stop_mlx_server()

    async def on_valves_updated(self):
        """Handle updates to the pipeline configuration."""
        self.update_valves()
        self.models = self.get_mlx_models()
        self.start_mlx_server(self.valves.MLX_DEFAULT_MODEL)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Process a request through the MLX pipeline."""
        logging.info(f"pipe:{__name__}")

        # Switch model if necessary
        if model_id != self.current_model:
            model_name = next((model['name'] for model in self.models if model['id'] == model_id), self.valves.MLX_DEFAULT_MODEL)
            self.start_mlx_server(model_name)

        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        max_tokens = body.get("max_tokens", 4096)
        temperature = body.get("temperature", 0.8)
        repeat_penalty = body.get("repeat_penalty", 1.0)

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": repeat_penalty,
            "stop": self.stop_sequence,
            "stream": body.get("stream", False),
        }
        
        # Request usage statistics in streaming mode for Open WebUI info panel
        if body.get("stream", False):
            payload["stream_options"] = {"include_usage": True}

        try:
            start_time = time.time()
            
            r = requests.post(
                url, headers=headers, json=payload, stream=body.get("stream", False)
            )

            r.raise_for_status()

            if body.get("stream", False):
                return self.process_stream(r.iter_lines(), start_time)
            else:
                result = r.json()
                elapsed_time = time.time() - start_time
                
                if 'choices' in result and len(result['choices']) > 0:
                    if 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                        content = result['choices'][0]['message']['content']
                        result['choices'][0]['message']['content'] = self.clean_response(content)
                
                # Add timing information to usage statistics
                if 'usage' in result:
                    completion_tokens = result['usage'].get('completion_tokens', 0)
                    if elapsed_time > 0 and completion_tokens > 0:
                        tokens_per_second = completion_tokens / elapsed_time
                        result['usage']['tokens_per_second'] = round(tokens_per_second, 2)
                        result['usage']['generation_time'] = round(elapsed_time, 3)
                    
                    logging.info(f"Response metadata: usage={result.get('usage')}, model={result.get('model')}")
                
                return result

        except Exception as e:
            return f"Error: {e}"