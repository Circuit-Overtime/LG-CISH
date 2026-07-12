# LG-CISH — Hiding Messages Inside Ordinary Photos, Without Touching Them

**LG-CISH** is a way to send a secret message by sharing a sequence of everyday
photographs — **without editing a single pixel of any photo**. The message isn't
written *into* the images; it's carried by **which images are sent and in what
order**.

Think of it like a bookshelf you and a friend both own, where every book has a known
position. Instead of scribbling a hidden note inside a book, you spell out your
message by *pointing at books in a certain order*. Anyone watching just sees you
looking at ordinary books — they can't tell a message was ever sent. LG-CISH does the
same thing with a shared set of 40 ordinary photos.

## Why this is interesting

- **Invisible by design.** Because the photos are never altered, tools that hunt for
  "tampered" images find nothing — they see normal pictures, so detectors do no better
  than a coin flip (~50%). Traditional methods that hide data *inside* pixels are
  caught almost every time.
- **Survives real chat apps.** When you send a photo through WhatsApp, Telegram, or
  social media, it gets shrunk and re-compressed. That destroys older hiding methods.
  LG-CISH recognises the photos by their *meaning* (using an AI image model called
  CLIP), so the message still comes back **perfectly** even after heavy compression,
  resizing, noise, or cropping.
- **Exact.** The recovered message matches the original bit-for-bit (0% error) on a
  normal channel.
- **Private.** The message is also compressed, encrypted (AES-256), and checksummed,
  so only the intended receiver — who shares the same photo set — can read it.

## How it works, in one picture

```
  "Meet me at the harbour."   →   a sequence of ordinary, unmodified photos
                                   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ...
                                   │ 🏔  │ │ 🌇  │ │ 🐕  │ │ 🚲  │
                                   └─────┘ └─────┘ └─────┘ └─────┘
        send them over any chat app  ─────────────────────────►
                                   the receiver looks at each photo,
                                   figures out which one it is, and
                                   reads the message back exactly.
```

The **order and identity** of the photos are the message. Nothing is hidden in the
pixels, so there's nothing to detect.

## What's in this repository

| Folder | What it is |
|--------|------------|
| `paper/` | The research paper (LaTeX source + PDF) with the full method, math, and results |
| `LG-COSH/` | The code — encoder, decoder, and the evaluation scripts that produced every number and figure |
| `dataset/` | The **40 photos** that make up the shared codebook, plus how to cite them |

## Cite this work

If you use LG-CISH or its dataset, please cite the repository (see
[`dataset/CITATION.cff`](dataset/CITATION.cff)) and, for the images, the original
UCID, Kodak, and USC-SIPI sources listed in [`dataset/README.md`](dataset/README.md).

## License

See [`LICENSE`](LICENSE). In short: the **code and paper are free to use** (MIT), and
the **dataset images are for non-commercial research** and remain subject to the terms
of their original sources (UCID, Kodak, USC-SIPI).

---

*LG-CISH = Language-Guided Coverless Image Steganography via Hashing.*
