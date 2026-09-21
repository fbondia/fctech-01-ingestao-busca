"""Interface de chat no terminal."""

from __future__ import annotations

from search import search_prompt


EXIT_COMMANDS = {"sair", "exit", "quit", ":q"}

def main():
    print("Faça sua pergunta sobre o PDF (digite 'sair' para encerrar).")

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté logo!")
            return

        if question.lower() in EXIT_COMMANDS:
            print("Até logo!")
            return
        if not question:
            print("Digite uma pergunta ou 'sair' para encerrar.")
            continue

        try:
            answer = search_prompt(question)
        except Exception as exc:
            print(f"ERRO: não foi possível responder: {exc}")
            continue

        print(f"RESPOSTA: {answer}")

if __name__ == "__main__":
    main()
