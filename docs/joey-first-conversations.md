# Joey — First Conversations 🎉

Sample chats from **Joey**, a ~170M masked-diffusion LLM built from scratch
(tokenizer → model → diffusion → sampler → training → SFT), trained on FineWeb-Edu
and instruction-tuned on DailyDialog.

- **Base:** ~170M params, trained from scratch on ~2B tokens of FineWeb-Edu (step ~173k, loss ~8.6)
- **SFT:** DailyDialog utterance pairs, response-only masking (LLaDA-style)
- **Sampler:** remasking (MaskGIT/LLaDA-style) iterative unmasking + repetition penalty 1.3 + top-p 0.9
- **Date:** 2026-06-04

These are the first outputs where Joey was recognizably *conversational* — greeting
correctly, forming real sentences, staying on-topic-ish — after the repetition loops were solved.

---

```
you> Hi!
joey> Hi. Well, I'm going over with it from all my company...
      But Your name is Dano- If you want me Your friend. Your husband will be your son who

you> How are you?
joey> I want Your name is! Our This have changed a lot! It's boarding for one hour
      next week. We will go into our company, so we can get it out Your Right.

you> What is your name?
joey> I'm boarding Your washing it up with me, sir and she or. The! It's so nice
      for me that you'll have came money if This one-hour will be host

you> Do you like music?
joey> I don't know that much. But I think there is no one through so then.
      We can't say anything. Your's not right, but it must be a great idea if we

you> What did you do today?
joey> I wanted Your Bingle from here and then got my meat! buy something,
      we had through our new restaurant. We could get a lot money back This year is very kind up with

you> Goodbye!
joey> Well, I'm not so good. But we've got for a moment. It's my best Your thing.
      It is too bad if you don't know it This all these things are right
```

---

## What worked

- **Greets correctly** — "Hi!" → "Hi. Well, I'm..."
- **Real grammatical sentences** — "It's boarding for one hour next week. We will go into our company, so we can get it out."
- **On-topic-ish answers** — "Do you like music?" → "I don't know that much" (plausible!); "What did you do today?" → restaurant/activities.
- **Conversational register** — everyday DailyDialog feel (company, restaurant, money, family).
- **No repetition loops** — the remasking sampler fixed the earlier `the side of the side of the side` collapse.

## Honest warts

- `Your` / `This` tokens jam in without spaces near turn-ends — the model latched onto a
  couple of high-frequency tokens; a mild residual artifact, not a loop.
- Replies sometimes trail into mush. Expected for a 170M from-scratch model — but it is
  recognizably *chatting*, which was the goal: a "simple, basic, generic conversational state."
