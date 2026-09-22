"""
VitaMind Mira V5.2
Bilingual terminal runner.

Run from project root:
    python run_mira.py
"""

from vitamind.mira.agent import MiraAgent


def print_header():
    print()
    print("=" * 64)
    print("                     VitaMind")
    print("                       MIRA V5.2")
    print("=" * 64)
    print()
    print("Mental-health screening and assessment prototype")
    print("English / العربية — you may switch languages at any time.")
    print()
    print("Commands:")
    print("  exit  - leave Mira")
    print("  quit  - leave Mira")
    print()


def main():
    print_header()

    mira = MiraAgent(debug=False)

    session, opening = mira.create_session(
        age_group="adult",
        language="auto",
        country_context="UAE",
    )

    print("MIRA:")
    print(opening.text)
    print()

    while not session.complete:
        try:
            user_text = input("YOU: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended.")
            break

        if user_text.lower() in {"exit", "quit"}:
            print("\nMIRA:")
            print("Thanks for speaking with me. Take care. / شكراً لتحدثك معي. مع السلامة.")
            break

        if not user_text:
            print("\nMIRA:")
            print(
                "You can answer naturally in English or Arabic. "
                "/ يمكنك الإجابة بطريقتك الطبيعية بالعربية أو الإنجليزية."
            )
            print()
            continue

        try:
            reply = mira.respond(session, user_text)
        except Exception as exc:
            print()
            print("SYSTEM ERROR:")
            print(str(exc))
            print()
            continue

        print()
        print("MIRA:")
        print(reply.text)
        print()

        if reply.assessment_complete:
            print("-" * 64)

            if reply.pathway:
                print(f"PATHWAY: {reply.pathway}")

            if reply.match_strength:
                print(f"MATCH STRENGTH: {reply.match_strength}")

            if reply.recommended_test:
                print(f"NEXT STEP: {reply.recommended_test}")

            if reply.language:
                print(f"RESPONSE LANGUAGE: {reply.language}")

            print("-" * 64)
            print()

    print()
    print("VitaMind Mira session closed.")


if __name__ == "__main__":
    main()
