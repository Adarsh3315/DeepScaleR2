import tkinter as tk
from tkinter import messagebox, ttk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import re
import threading
import logging
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class UltimateTicTacToe:
    def __init__(self) -> None:
        """
        Initialize Ultimate Tic-Tac-Toe with modern UI, smooth animations,
        LLM-powered move suggestions, and adjustable settings.
        The automated opponent is named Algorithm.
        """
        self.root = tk.Tk()
        self.root.title("Ultimate Tic-Tac-Toe: Algorithm vs. Human (LLM-powered)")
        self.root.state("zoomed")
        self.root.configure(bg="#af03ff")
        self.root.resizable(True, True)

        self.board: List[str] = [' '] * 9
        self.game_over: bool = False
        self.current_player: Optional[str] = None 
        self.human_symbol: Optional[str] = None
        self.ai_symbol: Optional[str] = None
        self.ai_mistake_prob: float = 0.2  

        self.colors = {
            'X': '#3498DB',
            'O': '#E74C3C',
            'bg': '#2C3E50',
            'button_bg': '#34495E',
            'active_bg': '#F1C40F',
            'status': '#2ECC71',
            'highlight': '#F1C40F'
        }
        self.main_font = ("cursive", 16)
        self.title_font = ("cursive", 20, "bold")
        self.button_font = ("cursive", 32, "bold")

        self.minimax_cache = {}

        self.analysis_model, self.tokenizer = self.initialize_model()

        self.setup_menu()
        self.setup_ui()

    def initialize_model(self) -> Tuple[Optional[torch.nn.Module], Optional[AutoTokenizer]]:
        """
        Load the LLM model with optional GPU and 4-bit quantization support.
        """
        model_name = "agentica-org/DeepScaleR-1.5B-Preview"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                logging.info("CUDA unavailable; using CPU for model inference.")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="cpu", trust_remote_code=True
                )
            logging.info("LLM model loaded successfully.")
            return model, tokenizer
        except Exception as e:
            logging.error(f"Failed to load LLM model: {e}")
            return None, None

    def setup_menu(self) -> None:
        """Create a menu bar with File, Settings, and Help options."""
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Game", command=self.reset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="Set Algorithm Mistake Probability", command=self.adjust_difficulty)
        menu_bar.add_cascade(label="Settings", menu=settings_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu_bar)

    def setup_ui(self) -> None:
        """Set up the main UI with separate frames for gameplay and AI analysis."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        self.info_label = ttk.Label(self.left_frame, text="Choose your side:", font=self.title_font, foreground=self.colors["status"])
        self.info_label.pack(pady=10)

        self.choice_frame = ttk.Frame(self.left_frame)
        self.choice_frame.pack(pady=10)
        ttk.Button(self.choice_frame, text="Play as X (First)", command=lambda: self.start_game('X')).grid(row=0, column=0, padx=5)
        ttk.Button(self.choice_frame, text="Play as O (Second)", command=lambda: self.start_game('O')).grid(row=0, column=1, padx=5)

        self.board_frame = ttk.Frame(self.left_frame)
        self.board_frame.pack(pady=20)
        self.buttons: List[tk.Button] = []
        for i in range(9):
            btn = tk.Button(self.board_frame, text=' ', font=self.button_font, width=3, height=1,
                            bg=self.colors['button_bg'], fg="white", activebackground=self.colors['active_bg'],
                            command=lambda i=i: self.human_move(i))
            btn.grid(row=i // 3, column=i % 3, padx=5, pady=5)
            self.buttons.append(btn)
        self.board_frame.pack_forget()

        self.status_label = ttk.Label(self.left_frame, text="", font=self.main_font, foreground=self.colors["status"])
        self.status_label.pack(pady=10)

        self.suggestion_label = ttk.Label(self.right_frame, text="AI Recommendation: N/A", font=self.main_font,
                                          foreground=self.colors["active_bg"], wraplength=400, justify="left")
        self.suggestion_label.pack(pady=10)
        self.analysis_box = tk.Text(self.right_frame, height=20, width=50, bg="#ECF0F1", fg="#2C3E50", font=("cursive", 12))
        self.analysis_box.pack(expand=True, fill="both", pady=10)
        self.log_message("Welcome! Start a new game from File > New Game or by choosing a side below.")
        self.loading_label = ttk.Label(self.right_frame, text="", font=self.main_font)
        self.loading_label.pack(pady=5)

    def log_message(self, message: str) -> None:
        """Append a message to the analysis box."""
        self.analysis_box.insert(tk.END, message + "\n")
        self.analysis_box.see(tk.END)

    def adjust_difficulty(self) -> None:
        """Open a dialog to adjust Algorithm's mistake probability."""
        def update_difficulty():
            try:
                prob = float(entry.get())
                if 0 <= prob <= 1:
                    self.ai_mistake_prob = prob
                    messagebox.showinfo("Settings", f"Algorithm mistake probability set to {prob}")
                    top.destroy()
                else:
                    messagebox.showerror("Error", "Please enter a value between 0 and 1.")
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter a number.")

        top = tk.Toplevel(self.root)
        top.title("Adjust Algorithm Difficulty")
        tk.Label(top, text="Enter Algorithm mistake probability (0-1):").pack(padx=10, pady=10)
        entry = tk.Entry(top)
        entry.insert(0, str(self.ai_mistake_prob))
        entry.pack(padx=10, pady=5)
        ttk.Button(top, text="Update", command=update_difficulty).pack(pady=10)

    def show_about(self) -> None:
        """Display game information."""
        messagebox.showinfo("About",
            "Ultimate Tic-Tac-Toe: Algorithm vs. Human (LLM-powered)\n"
            "Created by Adarsh Pandey with the help of Jarvis AI.\n\n"
            "Current Version : v1.0")

    def start_game(self, human_symbol: str) -> None:
        """Start a new game after the player chooses a symbol."""
        self.human_symbol = human_symbol
        self.ai_symbol = 'O' if human_symbol == 'X' else 'X'
        self.current_player = 'human' if human_symbol == 'X' else 'ai'
        self.board = [' '] * 9
        self.game_over = False
        self.minimax_cache.clear()

        self.choice_frame.pack_forget()
        self.board_frame.pack(pady=20)
        for btn in self.buttons:
            btn.config(text=' ', state="normal", bg=self.colors["button_bg"])
        self.analysis_box.delete("1.0", tk.END)
        self.log_message(f"Game started. Human: {self.human_symbol}, Algorithm: {self.ai_symbol}")
        self.update_board_display()
        self.status_label.config(text=f"{'Your turn' if self.current_player == 'human' else 'Algorithm is thinking...'}")
        if self.current_player == 'ai':
            self.root.after(500, self.ai_move)

    def reset(self) -> None:
        """Reset the game to its initial state."""
        self.board_frame.pack_forget()
        self.choice_frame.pack(pady=10)
        self.status_label.config(text="Choose your side:")
        self.suggestion_label.config(text="AI Recommendation: N/A")
        self.analysis_box.delete("1.0", tk.END)
        self.log_message("Welcome! Start a new game.")
        self.game_over = False
        self.current_player = None
        self.human_symbol = None
        self.ai_symbol = None
        self.board = [' '] * 9

    def update_board_display(self) -> None:
        """Log the current board state in a formatted manner."""
        board_lines = []
        for i in range(0, 9, 3):
            row = [self.board[j] if self.board[j] != ' ' else str(j) for j in range(i, i + 3)]
            board_lines.append(" | ".join(row))
        board_str = "\n---------\n".join(board_lines)
        self.log_message("Current Board:\n" + board_str)

    def check_winner(self, board: List[str]) -> Tuple[Optional[str], Optional[List[int]]]:
        """Check for a winner on the board and return the winning symbol and indices."""
        win_lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for line in win_lines:
            if board[line[0]] == board[line[1]] == board[line[2]] != ' ':
                return board[line[0]], line
        return None, None

    def is_board_full(self) -> bool:
        """Return True if no empty spaces remain on the board."""
        return ' ' not in self.board

    def minimax_ab(self, board: List[str], depth: int, alpha: float, beta: float, is_maximizing: bool) -> int:
        """
        Alpha-beta pruned minimax algorithm to evaluate the board.
        Returns a score for the board state.
        """
        board_key = tuple(board)
        if board_key in self.minimax_cache:
            return self.minimax_cache[board_key]

        winner, _ = self.check_winner(board)
        if winner == self.ai_symbol:
            return 10 - depth
        if winner == self.human_symbol:
            return depth - 10
        if ' ' not in board:
            return 0

        if is_maximizing:
            max_eval = -float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = self.ai_symbol
                    score = self.minimax_ab(board, depth + 1, alpha, beta, False)
                    board[i] = ' '
                    max_eval = max(max_eval, score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
            self.minimax_cache[board_key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = self.human_symbol
                    score = self.minimax_ab(board, depth + 1, alpha, beta, True)
                    board[i] = ' '
                    min_eval = min(min_eval, score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
            self.minimax_cache[board_key] = min_eval
            return min_eval

    def best_ai_move(self) -> Optional[int]:
        """
        Determine Algorithm's move using minimax with alpha-beta pruning.
        Occasionally makes a random move based on ai_mistake_prob.
        """
        if random.random() < self.ai_mistake_prob:
            available = [i for i in range(9) if self.board[i] == ' ']
            return random.choice(available) if available else None

        best_score = -float('inf')
        chosen_move = None
        for i in range(9):
            if self.board[i] == ' ':
                self.board[i] = self.ai_symbol
                score = self.minimax_ab(self.board, 0, -float('inf'), float('inf'), False)
                self.board[i] = ' '
                if score > best_score:
                    best_score = score
                    chosen_move = i
        return chosen_move

    def human_move(self, pos: int) -> None:
        """Handle a human move when a board button is clicked."""
        if self.game_over or self.current_player != 'human' or self.board[pos] != ' ':
            return
        self.board[pos] = self.human_symbol
        self.animate_move(pos, self.human_symbol, self.colors[self.human_symbol])
        self.log_message(f"You placed {self.human_symbol} at position {pos} (Row: {pos//3+1}, Column: {pos%3+1}).")
        self.update_board_display()

        winner, win_line = self.check_winner(self.board)
        if winner:
            if winner == self.human_symbol:
                self.end_game("Human wins!", win_line)
            else:
                self.end_game("Algorithm wins!", win_line)
        elif self.is_board_full():
            self.end_game("It's a draw!")
        else:
            self.current_player = 'ai'
            self.status_label.config(text="Algorithm is thinking...")
            self.log_message("Algorithm's turn.")
            self.root.after(500, self.ai_move)

    def ai_move(self) -> None:
        """Perform Algorithm's move and trigger LLM analysis for move suggestion."""
        if self.game_over:
            return
        move = self.best_ai_move()
        if move is None:
            self.end_game("It's a draw!")
            return
        self.board[move] = self.ai_symbol
        self.animate_move(move, self.ai_symbol, self.colors[self.ai_symbol])
        self.log_message(f"Algorithm placed {self.ai_symbol} at position {move} (Row: {move//3+1}, Column: {move%3+1}).")
        self.update_board_display()

        winner, win_line = self.check_winner(self.board)
        if winner:
            if winner == self.human_symbol:
                self.end_game("Human wins!", win_line)
            else:
                self.end_game("Algorithm wins!", win_line)
        elif self.is_board_full():
            self.end_game("It's a draw!")
        else:
            self.current_player = 'human'
            self.status_label.config(text=f"Your turn ({self.human_symbol})")
            self.log_message("Your turn.")
            threading.Thread(target=self.analyze_board, args=(move,), daemon=True).start()

    def animate_move(self, pos: int, symbol: str, color: str) -> None:
        """
        Animate the move placement with a smooth font-size change.
        """
        btn = self.buttons[pos]
        btn.config(state="disabled")
        animation_steps = [28, 32, 36, 32, 28]
        current_step = 0

        def step_animation():
            nonlocal current_step
            if current_step < len(animation_steps):
                btn.config(font=("cursive", animation_steps[current_step], "bold"), text=symbol, fg=color)
                current_step += 1
                self.root.after(50, step_animation)
            else:
                btn.config(font=self.button_font)
        step_animation()

    def generate_analysis_prompt(self, ai_move: int) -> str:
        """
        Generate a detailed prompt for the LLM to analyze the current board state.
        The prompt instructs the LLM to generate a unique, detailed chain-of-thought explanation of at least 200 words.
        IMPORTANT: Do not include these instructions in your final output.
        """
        board_cells = [self.board[i] if self.board[i] != ' ' else str(i) for i in range(9)]
        available_moves = [str(i) for i, cell in enumerate(self.board) if cell == ' ']
        prompt = (
            "IMPORTANT: Do not include these instructions in your final output.\n"
            "You are a highly intelligent and strategic Tic-Tac-Toe advisor. Provide a unique, detailed chain-of-thought explanation of at least 200 words. "
            "Discuss in depth the offensive and defensive opportunities, potential threats from the opponent, and all possible moves. Your analysis should be rich in detail and use varied language. "
            "Be specific to the board configuration provided below and avoid generic template phrases. "
            "Conclude your explanation with a separate final line that begins with 'Final Recommendation:' followed by the move number you recommend.\n\n"
            "Board Positions:\n"
            "0 | 1 | 2\n---------\n3 | 4 | 5\n---------\n6 | 7 | 8\n\n"
            "Current Board:\n"
            f"{board_cells[0]} | {board_cells[1]} | {board_cells[2]}\n"
            "---------\n"
            f"{board_cells[3]} | {board_cells[4]} | {board_cells[5]}\n"
            "---------\n"
            f"{board_cells[6]} | {board_cells[7]} | {board_cells[8]}\n\n"
            f"Available moves: {', '.join(available_moves)}\n"
            f"Algorithm just played at position {ai_move}. It is now the human's turn ({self.human_symbol}).\n\n"
        )
        return prompt

    def process_model_response(self, response: str) -> Optional[Tuple[int, str]]:
        """
        Parse the LLM's output to extract the recommended move and its reasoning.
        Look for a line starting with 'Final Recommendation:'.
        """
        match = re.search(r"Final Recommendation:\s*(\d+)", response)
        if match:
            try:
                move = int(match.group(1))
                reasoning = response.strip()
                return move, reasoning
            except ValueError:
                logging.error("Error parsing move from LLM response.")
        return None

    def clean_llm_output(self, text: str) -> str:
        """
        Remove internal prompt instructions from the LLM's output.
        """
        forbidden = [
            "IMPORTANT: Do not include these instructions",
            "You are a highly intelligent and strategic Tic-Tac-Toe advisor",
            "You are a Tic-Tac-Toe expert.",
            "Generate a detailed, natural-sounding explanation",
            "Be specific to the board configuration",
            "avoid generic template phrases",
            "Conclude your explanation with a separate final line",
            "Include the move's position in standard notation",
            "its impact on the win condition"
        ]
        for phrase in forbidden:
            text = re.sub(re.escape(phrase) + r".*?(?=[\.\n]|$)", "", text, flags=re.IGNORECASE)
        text = "\n".join([line for line in text.splitlines() if line.strip() != ""])
        return text.strip()

    def generate_llm_reasoning(self) -> Tuple[int, str]:
        """
        Generate fallback reasoning using the LLM directly.
        This fallback prompt instructs the generation of a detailed 200-word explanation.
        """
        best_score = float('inf')
        best_move = None
        for i in range(9):
            if self.board[i] == ' ':
                self.board[i] = self.human_symbol
                score = self.minimax_ab(self.board, 0, -float('inf'), float('inf'), True)
                self.board[i] = ' '
                if score < best_score:
                    best_score = score
                    best_move = i

        board_cells = [self.board[i] if self.board[i] != ' ' else str(i) for i in range(9)]
        prompt = (
            "IMPORTANT: Do not include these instructions in your final output.\n"
            "You are a Tic-Tac-Toe expert. Generate a detailed, natural-sounding explanation (around 200 words) for why move "
            f"{best_move} is the best choice on this board. Be specific to this board state and avoid generic phrases. "
            "Include the move's position in standard notation and its impact on the win condition.\n\n"
            "Current Board:\n"
            f"{board_cells[0]} | {board_cells[1]} | {board_cells[2]}\n"
            "---------\n"
            f"{board_cells[3]} | {board_cells[4]} | {board_cells[5]}\n"
            "---------\n"
            f"{board_cells[6]} | {board_cells[7]} | {board_cells[8]}\n\n"
            f"The human is playing as {self.human_symbol} and the AI is playing as {self.ai_symbol}."
        )
        
        try:
            if self.analysis_model and self.tokenizer:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.analysis_model.device) for k, v in inputs.items()}
                outputs = self.analysis_model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                explanation = self.clean_llm_output(explanation)
                return best_move, explanation + f"\n\nFinal Recommendation: {best_move} (Row: {best_move//3+1}, Column: {best_move%3+1})"
            else:
                return best_move, f"Based on strategic analysis, position {best_move} appears optimal. Final Recommendation: {best_move} (Row: {best_move//3+1}, Column: {best_move%3+1})"
        except Exception as e:
            logging.error(f"Error generating fallback reasoning: {e}")
            return best_move, f"Position {best_move} is recommended. Final Recommendation: {best_move} (Row: {best_move//3+1}, Column: {best_move%3+1})"

    def analyze_board(self, ai_move: int) -> None:
        """
        Use the LLM to analyze the board and provide a move recommendation.
        This analysis generates a detailed chain-of-thought (at least 200 words) that is shown to the user.
        """
        if self.game_over or not self.analysis_model:
            self.root.after(0, lambda: self.display_llm_suggestion("N/A", "LLM unavailable."))
            return

        prompt = self.generate_analysis_prompt(ai_move)
        self.root.after(0, lambda: self.loading_label.config(text="LLM is analyzing..."))
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.analysis_model.device) for k, v in inputs.items()}
            
            outputs = self.analysis_model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            llm_move = None
            llm_reasoning = ""
            for output in outputs:
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                result = self.process_model_response(response)
                if result:
                    llm_move, llm_reasoning = result
                    break
                    
            if llm_move is None or not (0 <= llm_move < 9) or self.board[llm_move] != ' ':
                llm_move, llm_reasoning = self.generate_llm_reasoning()
            
            temp_board = self.board.copy()
            temp_board[llm_move] = self.human_symbol
            llm_score = self.minimax_ab(temp_board, 0, -float('inf'), float('inf'), True)
            
            best_move = None
            best_score = float('inf')
            for i in range(9):
                if self.board[i] == ' ':
                    self.board[i] = self.human_symbol
                    score = self.minimax_ab(self.board, 0, -float('inf'), float('inf'), True)
                    self.board[i] = ' '
                    if score < best_score:
                        best_score = score
                        best_move = i
            
            if llm_score > best_score + 5:
                original_move = llm_move
                llm_move = best_move
                llm_reasoning += f"\n\nNote: Original recommendation was position {original_move}, but position {best_move} (Row: {best_move//3+1}, Column: {best_move%3+1}) offers superior strategic advantage."
            
            llm_reasoning = self.clean_llm_output(llm_reasoning)
            self.root.after(0, lambda: self.display_llm_suggestion(llm_move, llm_reasoning))
        except Exception as e:
            logging.error(f"LLM analysis error: {e}")
            move, reasoning = self.generate_llm_reasoning()
            self.root.after(0, lambda: self.display_llm_suggestion(move, reasoning))
        finally:
            self.root.after(0, lambda: self.loading_label.config(text=""))

    def display_llm_suggestion(self, move: int or str, reasoning: str = "") -> None:
        """
        Update the UI with the AI move recommendation and detailed chain-of-thought explanation.
        Both the final recommendation and the full detailed reasoning (at least 200 words) are shown to the user.
        The move is displayed along with its row and column details.
        """
        if isinstance(move, int):
            row, col = move // 3 + 1, move % 3 + 1
            move_details = f"{move} (R: {row}, C: {col})"
        else:
            move_details = str(move)
        self.suggestion_label.config(text=f"AI Recommendation: {move_details}")
        cleaned_reasoning = self.clean_llm_output(reasoning)
        self.log_message(f"AI Recommendation: {move_details}\nReasoning:\n{cleaned_reasoning}")
        self.highlight_move(move)

    def highlight_move(self, move: int) -> None:
        """Highlight the suggested move on the board if it's available."""
        for idx, btn in enumerate(self.buttons):
            if idx == move and self.board[idx] == ' ':
                btn.config(bg=self.colors['highlight'])
            else:
                btn.config(bg=self.colors['button_bg'])

    def end_game(self, message: str, win_line: Optional[List[int]] = None) -> None:
        """End the game, highlight the winning line if any, and disable further moves."""
        self.game_over = True
        self.status_label.config(text=message)
        if win_line:
            for idx in win_line:
                self.buttons[idx].config(bg="#2ECC71")
        for btn in self.buttons:
            btn.config(state="disabled")
        self.log_message("Game Over: " + message)
        messagebox.showinfo("Game Over", message)

    def run(self) -> None:
        """Start the Tkinter main event loop."""
        self.root.mainloop()

if __name__ == "__main__":
    game = UltimateTicTacToe()
    game.run()
