use std::collections::hash_map::Entry;
use std::collections::HashMap;

use peekmore::{PeekMore, PeekMoreIterator};


struct Node {
    terminal: bool,
    children: HashMap<String, Node>
}

impl Node {
    pub fn new() -> Self {
        Self{terminal: false, children: HashMap::new()}
    }

    pub fn insert<T: Iterator<Item=String>>(&mut self, sequence: &mut T) {
        match sequence.next() {
            None => self.terminal = true,
            Some(x) => {
                match self.children.entry(x) {
                    Entry::Occupied(mut e) => {
                        e.get_mut().insert(sequence);
                    }
                    Entry::Vacant(mut e) => {
                        e.insert(Node::new()).insert(sequence);
                    }
                }
            }
        }
    }

    pub fn max_depth(&self) -> usize {
        let x = self.children
            .values()
            .map(|n| n.max_depth())
            .max();
        match x {
            None => 1,
            Some(depth) => depth + 1
        }
    }

    pub fn longest_match<T>(&self, sequence: &mut PeekMoreIterator<T>) -> Option<usize>
    where
        T: Iterator<Item=String>
    {
         match sequence.peek() {
            None => {
                // End of input -> No need to check further.
                if self.terminal { Some(1) } else { None }
            }
            Some(x) => {
                match self.children.get(x) {
                    None => {
                        // No children -> No need to check further
                        if self.terminal { Some(1) } else { None }
                    }
                    Some(node) => {
                        match node.longest_match(sequence.advance_cursor()) {
                            None => {
                                // No deeper match; are we a match?
                                sequence.reset_cursor();
                                if self.terminal { Some(1) } else { None }
                            }
                            Some(d) => {
                                // Deeper match; propagate match length
                                sequence.reset_cursor();
                                Some(d + 1)
                            }
                        }
                    }
                }
            }
        }
    }
}

pub struct WordTrie {
    root: Node
}

impl WordTrie {
    pub fn new(words: Vec<Vec<String>>) -> Self {
        let mut root = Node::new();
        for sequence in words {
            root.insert(&mut sequence.into_iter());
        }
        Self{root}
    }

    pub fn replace_substrings_with(&self,
                                   document: Vec<String>,
                                   replacement: &Vec<String>) -> Vec<String> {
        let mut stack = Vec::with_capacity(document.len());
        let mut stream = document.into_iter().peekmore();
        loop {
            match self.root.longest_match(&mut stream) {
                None => {
                    // Push next symbol to results
                    match stream.next() {
                        None => break,
                        Some(x) => stack.push(x)
                    }
                }
                Some(length) => {
                    // Found a match; add replacement
                    stack.extend_from_slice(replacement.as_slice());
                    // Skip symbols
                    for _ in 0..length - 1 {
                        stream.next();
                    }
                }
            }
        }
        stack
    }
}