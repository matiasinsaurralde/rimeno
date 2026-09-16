package dotenv

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParse(t *testing.T) {
	in := `
# a comment
export OPENAI_API_KEY=sk-abc123
OPENAI_BASE_URL = https://openrouter.ai/api/v1
QUOTED="a b c"
SINGLE='x y'
ESCAPED="line1\nline2"
1BAD=nope
EMPTY=
`
	got, err := Parse(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"OPENAI_API_KEY":  "sk-abc123",
		"OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
		"QUOTED":          "a b c",
		"SINGLE":          "x y",
		"ESCAPED":         "line1\nline2",
		"EMPTY":           "",
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s = %q, want %q", k, got[k], v)
		}
	}
	if _, ok := got["1BAD"]; ok {
		t.Errorf("invalid key 1BAD should be skipped")
	}
}

func TestLoadFile_DoesNotOverride(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, ".env")
	if err := os.WriteFile(p, []byte("RIMENO_TEST_PRESENT=fromfile\nRIMENO_TEST_ABSENT=fromfile\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RIMENO_TEST_PRESENT", "fromenv") // already set → must win
	_ = os.Unsetenv("RIMENO_TEST_ABSENT")
	t.Cleanup(func() { _ = os.Unsetenv("RIMENO_TEST_ABSENT") })

	if err := LoadFile(p); err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv("RIMENO_TEST_PRESENT"); got != "fromenv" {
		t.Errorf("existing env overridden: %q, want fromenv", got)
	}
	if got := os.Getenv("RIMENO_TEST_ABSENT"); got != "fromfile" {
		t.Errorf("absent var not loaded: %q, want fromfile", got)
	}
}

func TestLoad_MissingIsNoError(t *testing.T) {
	dir := t.TempDir()
	p, err := Load(filepath.Join(dir, "does-not-exist.env"))
	if err != nil {
		t.Errorf("missing file should not error: %v", err)
	}
	if p != "" {
		t.Errorf("expected empty path for missing file, got %q", p)
	}
}
