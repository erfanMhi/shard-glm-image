#!/usr/bin/env python3
"""Generate prompts/study100.jsonl for the WAN-PP throughput study.

Writes 101 lines: the receipt prompt (code-000, "def quicksort(arr):") plus
100 new prompts in four categories (40 code, 20 prose, 20 reasoning,
20 instruct). Every prompt is validated before the file is written: ASCII
only, no trailing whitespace on any line, unique, code stubs 1 to 4 lines
and at most 60 words, other prompts 8 to 60 words, and no new prompt
contains the word "quicksort".

Usage: python3 make_prompts.py [output_path]
"""
import json
import os
import sys

RECEIPT = ("code-000", "code", "def quicksort(arr):")


def L(*lines):
    """Join stub lines with newlines (keeps the source readable)."""
    return "\n".join(lines)


CODE = [
    # Python (16)
    L("def binary_search(sorted_list, target):",
      '    """Return the index of target in sorted_list, or -1 if absent."""'),
    L("class LRUCache:",
      '    """A least-recently-used cache with a fixed capacity."""',
      "    def __init__(self, capacity):"),
    L('def parse_csv_line(line, delimiter=","):',
      '    """Split one CSV line into fields, honoring double-quoted fields."""'),
    L("def is_palindrome(s):",
      '    """Return True if s reads the same forwards and backwards, ignoring case and non-letters."""'),
    L("import numpy as np",
      "",
      "def softmax(x, axis=-1):",
      '    """Numerically stable softmax along the given axis."""'),
    L("def read_lines_with_numbers(path):",
      '    """Yield (line_number, line) pairs for each line in the file at path."""'),
    L("class Stack:",
      '    """A simple LIFO stack backed by a Python list."""'),
    L("import unittest",
      "",
      "class TestFibonacci(unittest.TestCase):",
      "    def test_small_values(self):"),
    L("def merge_sorted(a, b):",
      '    """Merge two sorted lists into one sorted list in O(len(a) + len(b)) time."""'),
    L("def word_frequencies(text):",
      '    """Return a dict mapping each lowercase word in text to its count."""'),
    L("def newton_sqrt(x, tol=1e-10):",
      '    """Approximate the square root of x with Newton\'s method."""'),
    L("class BinaryTreeNode:",
      "    def __init__(self, value, left=None, right=None):",
      "        self.value = value"),
    L("def flatten(nested):",
      '    """Flatten an arbitrarily nested list of lists into a single flat list."""'),
    L("def matrix_multiply(a, b):",
      '    """Multiply two matrices given as lists of lists; raise ValueError on a shape mismatch."""'),
    L("from dataclasses import dataclass",
      "",
      "@dataclass",
      "class Point:"),
    L("def tokenize_expression(expr):",
      '    """Split an arithmetic expression such as 3 + 4*(2 - 1) into a list of tokens."""'),
    # JavaScript / TypeScript (8)
    L("function debounce(fn, delayMs) {",
      "  // Return a debounced version of fn that waits delayMs after the last call."),
    L("function deepClone(obj) {",
      "  // Return a deep copy of obj, handling arrays, plain objects, and Dates."),
    L("async function fetchJsonWithRetry(url, retries = 3) {"),
    L("class EventEmitter {",
      "  constructor() {",
      "    this.listeners = new Map();",
      "  }"),
    L("// Return an array of the k most frequent elements in nums.",
      "function topKFrequent(nums, k) {"),
    L("type User = { id: number; name: string; email: string };",
      "",
      "function groupByDomain(users: User[]): Map<string, User[]> {"),
    L("export function parseQueryString(qs: string): Record<string, string> {",
      "  // Parse 'a=1&b=hello%20world' into an object with decoded values."),
    L("function isValidParentheses(s) {",
      "  // Return true if every bracket in s is closed in the correct order.",
      "  const stack = [];"),
    # Rust (4)
    L("/// Return the nth Fibonacci number using an iterative loop.",
      "fn fibonacci(n: u32) -> u64 {"),
    L("use std::collections::HashMap;",
      "",
      "/// Count how many times each word occurs in the input text.",
      "fn word_count(text: &str) -> HashMap<String, usize> {"),
    L("/// A fixed-capacity ring buffer of f64 samples.",
      "pub struct RingBuffer {",
      "    data: Vec<f64>,"),
    L('/// Parse a line like "key=value" into a (key, value) pair, trimming whitespace.',
      "fn parse_key_value(line: &str) -> Option<(&str, &str)> {"),
    # Go (4)
    L("// Reverse returns s with its runes in reverse order.",
      "func Reverse(s string) string {"),
    L("// ReadLines reads the file at path and returns its lines without trailing newlines.",
      "func ReadLines(path string) ([]string, error) {"),
    L("// Queue is a FIFO queue of ints backed by a slice.",
      "type Queue struct {"),
    L("func TestMax(t *testing.T) {",
      "\tcases := []struct{ in []int; want int }{"),
    # C / C++ (4)
    L("/* Return the length of the longest common prefix of the two strings. */",
      "size_t common_prefix_len(const char *a, const char *b) {"),
    L("#include <stdio.h>",
      "",
      "/* Read integers from stdin until EOF and print their sum. */",
      "int main(void) {"),
    L("template <typename T>",
      "class Matrix {",
      "public:",
      "    Matrix(size_t rows, size_t cols)"),
    L("// Swap the byte order of a 32-bit unsigned integer.",
      "uint32_t swap_endian(uint32_t x) {"),
    # SQL / Bash (4)
    L("-- Return each customer's name and total spend over the last 30 days, highest spend first.",
      "SELECT"),
    L("-- Table: employees(id, name, department, salary). Find the second highest salary in each department.",
      "WITH ranked AS ("),
    L("#!/usr/bin/env bash",
      "# Back up every .log file in the given directory into a dated tar.gz archive.",
      "set -euo pipefail"),
    L("#!/bin/bash",
      "# Print the 10 largest files under the current directory with human-readable sizes.",
      "find . -type f"),
]

PROSE = [
    "The first time I saw the ocean, I was eleven years old and convinced that",
    "Product description: The Meridian desk lamp combines a weighted brass base with a fully articulated arm, so",
    L("Dear Ms. Alvarez,",
      "",
      "Thank you for your letter of March 3rd regarding the community garden proposal. After discussing it with the council,"),
    "The lighthouse keeper had not spoken to another person in forty days, and he had begun to",
    "The Great Barrier Reef, located off the coast of Queensland, Australia, is the world's largest coral reef system. It",
    L("Essay: Why cities should plant more trees",
      "",
      "Every summer, the difference between a shaded street and an exposed one"),
    "Once upon a time, in a village at the edge of a forest that no map agreed on, there lived",
    "The printing press, introduced to Europe by Johannes Gutenberg around 1440, transformed",
    L("To whom it may concern,",
      "",
      "I am writing to recommend Daniel Okafor for the position of senior data engineer. In the three years"),
    "Introducing the Northwind travel backpack: forty liters of thoughtfully divided space, a laptop sleeve that",
    "The detective set down her coffee, looked at the photograph a second time, and realized",
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into",
    "On the morning the river froze, the whole town came out to look, because",
    L("A short history of the bicycle",
      "",
      "The earliest two-wheeled machines of the 1810s had no pedals; riders"),
    "Review: The new neighborhood bakery opened last Saturday, and by nine in the morning the line",
    L("Dear future me,",
      "",
      "If you are reading this, it means the time capsule survived and"),
    "The Silk Road was not a single road but a network of trade routes that",
    "My grandmother kept a notebook of every recipe she ever changed, and the margins",
    L("Blog post: What I learned from a year without a smartphone",
      "",
      "The first week was the hardest, mostly because"),
    "Mount Kilimanjaro, a dormant volcano in Tanzania, is the highest mountain in Africa. Its three volcanic cones",
]

REASONING = [
    "A train leaves the station at 9:00 am traveling 60 miles per hour. A second train leaves the same station at 10:30 am traveling 90 miles per hour on a parallel track. At what time does the second train catch up? Answer:",
    "Alice is older than Bob. Carol is younger than Bob but older than Dave. Who is the second youngest of the four? Answer:",
    "A shop sells pens at 3 for 2 dollars. How much do 18 pens cost, and how many pens can you buy with 10 dollars? Answer:",
    "Why does a metal spoon feel colder than a wooden spoon when both have been sitting in the same room? Explain step by step.",
    "You have a 3 liter jug and a 5 liter jug and unlimited water. How can you measure exactly 4 liters? Give the steps.",
    "A rectangle has a perimeter of 30 cm and its length is twice its width. What is its area in square centimeters? Answer:",
    "If all bloops are razzies and some razzies are lazzies, does it follow that some bloops are lazzies? Explain your reasoning.",
    "Why does ice float on water while most solids sink in their own liquid? Explain the mechanism in a few steps.",
    "You need to cook three dishes that take 20, 35, and 50 minutes and you have two burners. What is the shortest time to finish all three, and in what order? Answer:",
    "A bat and a ball cost 1.10 dollars in total. The bat costs one dollar more than the ball. How much does the ball cost? Answer:",
    "Three boxes are labeled apples, oranges, and mixed, but every label is wrong. You may draw one fruit from one box. How do you relabel all three correctly? Answer:",
    "Why do we see lightning before we hear the thunder from the same strike? Explain step by step.",
    "A bookshelf holds 120 books. Two fifths are fiction and a quarter of the rest are biographies. How many biographies are there? Answer:",
    "A farmer has chickens and cows with 30 heads and 74 legs in total. How many of each animal does the farmer have? Answer:",
    "Why does bread dough rise, and why does it stop rising once it is baked? Explain the two reasons in order.",
    "Plan a 6 hour workday that fits a 90 minute meeting, two 2 hour focus blocks, and a 30 minute lunch, with the meeting not first. Give the schedule.",
    "The sum of three consecutive even numbers is 78. What is the largest of the three? Answer:",
    "Why does a helium balloon rise in air but a balloon filled with your breath sinks? Explain the reasoning.",
    "Every knight tells the truth and every knave lies. Person A says: 'We are both knaves.' What are A and B? Answer:",
    "A recipe for 4 people needs 300 grams of rice. You are cooking for 10 people but only have 600 grams. How many people can you serve fully, and how much rice is left over? Answer:",
]

INSTRUCT = [
    "Write a haiku about a city waking up on a rainy morning.",
    "List five common mistakes people make when learning to cook, with one sentence on how to avoid each.",
    "Summarize the difference between a process and a thread in an operating system in three sentences.",
    "Give step-by-step instructions to change a flat bicycle tire using only a hand pump and a tire lever.",
    "Write a two-sentence product tagline for a reusable water bottle that keeps drinks cold for 24 hours.",
    "Explain to a ten year old how a vaccine trains the immune system, using one simple analogy.",
    "Draft a polite email declining a meeting invitation and proposing two alternative times next week.",
    "List the planets of the solar system in order from the sun, with one distinguishing fact about each.",
    "Write a limerick about a programmer who forgets to save their work.",
    "Give three tips for giving a clear five minute presentation to a non-technical audience.",
    "Summarize the plot of Romeo and Juliet in exactly four sentences.",
    "Translate the following sentence into French, Spanish, and German: 'The library closes at eight on weekdays.'",
    "Write a short motivational note, under 60 words, for a friend starting their first day at a new job.",
    "Explain the difference between weather and climate in plain language, then give one example of each.",
    "Create a simple weekly meal plan for one person with five dinners that share ingredients to reduce waste.",
    "Give step-by-step instructions to set up a new Git repository locally and push it to a remote for the first time.",
    "List six questions a person should ask before adopting a dog, and say why each one matters.",
    "Write a four line poem about the last leaf of autumn, without using the word 'fall'.",
    "Describe how to brew a good cup of pour-over coffee in five numbered steps.",
    "Compare electric and gas cars on cost, range, and maintenance in a short table followed by a one sentence recommendation.",
]

EXPECTED = {"code": 40, "prose": 20, "reasoning": 20, "instruct": 20}


def validate(rows):
    seen = set()
    counts = {}
    for pid, cat, prompt in rows:
        assert pid not in seen, f"duplicate id {pid}"
        seen.add(pid)
        assert prompt not in {r[2] for r in rows if r[0] != pid}, f"duplicate prompt {pid}"
        assert prompt.isascii(), f"non-ASCII in {pid}"
        assert "\u2014" not in prompt, f"em-dash in {pid}"
        assert prompt == prompt.rstrip(), f"trailing whitespace in {pid}"
        for line in prompt.split("\n"):
            assert line == line.rstrip(), f"trailing whitespace on a line of {pid}"
        if pid != "code-000":
            assert "quicksort" not in prompt.lower(), f"quicksort in {pid}"
        words = len(prompt.split())
        assert words <= 60, f"{pid} has {words} words"
        if cat == "code":
            nlines = prompt.count("\n") + 1
            assert 1 <= nlines <= 4, f"{pid} has {nlines} lines"
        else:
            assert words >= 8, f"{pid} has only {words} words"
        counts[cat] = counts.get(cat, 0) + 1
    assert counts == {"code": 41, "prose": 20, "reasoning": 20, "instruct": 20}, counts
    return counts


def build_rows():
    rows = [RECEIPT]
    for cat, items in (("code", CODE), ("prose", PROSE),
                       ("reasoning", REASONING), ("instruct", INSTRUCT)):
        assert len(items) == EXPECTED[cat], (cat, len(items))
        for i, prompt in enumerate(items, 1):
            rows.append((f"{cat}-{i:03d}", cat, prompt))
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "prompts", "study100.jsonl")
    rows = build_rows()
    counts = validate(rows)
    with open(out, "w", encoding="ascii") as fh:
        for pid, cat, prompt in rows:
            fh.write(json.dumps({"id": pid, "cat": cat, "prompt": prompt}) + "\n")
    print(f"wrote {len(rows)} lines to {os.path.normpath(out)}: {counts}")


if __name__ == "__main__":
    main()
