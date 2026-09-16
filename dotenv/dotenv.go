// Package dotenv loads environment variables from a .env file, as a local-dev
// convenience — e.g. keeping OPENAI_API_KEY / OPENAI_BASE_URL out of your shell
// history. It NEVER overrides variables already present in the process environment
// (the real environment always wins), so it is safe to call unconditionally at the
// start of a program.
//
//	dotenv.Load()                 // find and load the nearest .env (cwd upward), if any
//	dotenv.Load("/path/to/.env")  // load a specific file
//
// Format: `KEY=VALUE` per line; blank lines and lines starting with `#` are
// ignored; an optional leading `export ` is allowed; values may be single- or
// double-quoted (double quotes honor \n \t \r \" \\ escapes). It is stdlib-only.
package dotenv

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Load loads the first existing file among paths. With no paths, it searches for a
// .env from the current directory upward (stopping at a repository root, i.e. a
// directory containing .git). It returns the path actually loaded ("" if none was
// found) and never treats a missing file as an error.
func Load(paths ...string) (string, error) {
	if len(paths) == 0 {
		if p := find(); p != "" {
			paths = []string{p}
		}
	}
	for _, p := range paths {
		if p == "" {
			continue
		}
		if _, err := os.Stat(p); err != nil {
			continue
		}
		if err := LoadFile(p); err != nil {
			return "", err
		}
		return p, nil
	}
	return "", nil
}

// find walks up from the current working directory looking for a .env file. It
// stops at (and includes) a directory that contains a .git entry, or the filesystem
// root. It returns the path to the first .env found, or "".
func find() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		candidate := filepath.Join(dir, ".env")
		if st, err := os.Stat(candidate); err == nil && !st.IsDir() {
			return candidate
		}
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return "" // reached a repo root without finding .env; stop
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "" // filesystem root
		}
		dir = parent
	}
}

// LoadFile parses path and applies it to the environment (without overriding
// existing variables).
func LoadFile(path string) error {
	f, err := os.Open(path) // #nosec G304 -- path is the caller-chosen dotenv file to load
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	vars, err := Parse(f)
	if err != nil {
		return fmt.Errorf("dotenv %s: %w", path, err)
	}
	for k, v := range vars {
		if _, present := os.LookupEnv(k); !present {
			_ = os.Setenv(k, v)
		}
	}
	return nil
}

// Parse reads KEY=VALUE pairs from r. It is exported for testing/reuse; it does not
// touch the process environment.
func Parse(r io.Reader) (map[string]string, error) {
	out := map[string]string{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(strings.TrimRight(sc.Text(), "\r"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		eq := strings.IndexByte(line, '=')
		if eq <= 0 {
			continue // no key, or leading '=' — skip
		}
		key := strings.TrimSpace(line[:eq])
		if !validKey(key) {
			continue
		}
		out[key] = unquote(strings.TrimSpace(line[eq+1:]))
	}
	return out, sc.Err()
}

func validKey(k string) bool {
	if k == "" {
		return false
	}
	for i, r := range k {
		switch {
		case r >= 'A' && r <= 'Z', r >= 'a' && r <= 'z', r == '_':
		case r >= '0' && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}

// unquote strips matching surrounding quotes; double-quoted values honor common
// escapes. An unquoted value is returned as-is (already trimmed).
func unquote(v string) string {
	if len(v) >= 2 && v[0] == '"' && v[len(v)-1] == '"' {
		inner := v[1 : len(v)-1]
		repl := strings.NewReplacer(`\n`, "\n", `\r`, "\r", `\t`, "\t", `\"`, `"`, `\\`, `\`)
		return repl.Replace(inner)
	}
	if len(v) >= 2 && v[0] == '\'' && v[len(v)-1] == '\'' {
		return v[1 : len(v)-1]
	}
	return v
}
