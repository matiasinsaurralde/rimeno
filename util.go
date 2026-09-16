package rimeno

import (
	"crypto/rand"
	"encoding/hex"
)

// newID returns a short random identifier for spans and calls.
func newID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand should not fail; fall back to a fixed marker rather than panic.
		return "0000000000000000"
	}
	return hex.EncodeToString(b[:])
}
