use std::collections::hash_map::Entry;
use std::collections::HashMap;
use peekmore::{PeekMore, PeekMoreIterator};

struct Node {
    children: HashMap<char, Node>,
    is_terminal: bool
}

impl Node {
    pub fn new() -> Self {
        Self{children: HashMap::new(), is_terminal: false}
    }

    pub fn insert<T: Iterator<Item=char>>(&mut self, stream: &mut T) -> usize {
        match stream.next() {
            None => {
                self.is_terminal = true;
                0
            }
            Some(c) => {
                match self.children.entry(c) {
                    Entry::Occupied(mut e) => {
                        1 + e.get_mut().insert(stream)
                    }
                    Entry::Vacant(e) => {
                        1 + e.insert(Node::new()).insert(stream)
                    }
                }
            }
        }
    }

    pub fn longest_match<T>(&self, stream: &mut PeekMoreIterator<T>) -> Option<usize>
    where
        T: Iterator<Item=char>
    {
        match stream.peek() {
            None => {
                // End of input -> No need to check further.
                if self.is_terminal { Some(1) } else { None }
            }
            Some(x) => {
                match self.children.get(x) {
                    None => {
                        // No children -> No need to check further
                        if self.is_terminal { Some(1) } else { None }
                    }
                    Some(node) => {
                        match node.longest_match(stream.advance_cursor()) {
                            None => {
                                // No deeper match; are we a match?
                                stream.reset_cursor();
                                if self.is_terminal { Some(1) } else { None }
                            }
                            Some(d) => {
                                // Deeper match; propagate match length
                                stream.reset_cursor();
                                Some(d + 1)
                            }
                        }
                    }
                }
            }
        }
    }
}

pub struct CharTrie {
    root: Node,
    max_depth: usize
}

impl CharTrie {
    pub fn new(words: Vec<String>) -> Self {
        let mut root = Node::new();
        let mut max_depth = 0;
        for word in words {
            max_depth = max_depth.max(root.insert(&mut word.chars().into_iter()));
        }
        Self{root, max_depth}
    }

     pub fn replace_substrings_with(&self,
                                    document: String,
                                    replacement: &String) -> String {
        let mut stack = String::with_capacity(document.len());
        let mut stream = document.chars().into_iter().peekmore();
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
                    stack.push_str(replacement);
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